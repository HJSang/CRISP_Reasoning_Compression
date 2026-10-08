"""One full-checkpoint HF update using the pinned OPSD input and loss code.

This is an independent full-model diagnostic, not a run of the original TRL/vLLM
trainer and not a mapped HF-to-Megatron optimizer comparison. Uses four fresh
student trajectories with microbatch one and FP32 distribution arithmetic.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from miles.backends.training_utils.loss.hub.opsd_math import OPSDLossConfig, opsd_per_token_loss
from tests.opsd_reference import load_author_collator, load_author_input_builder, load_author_loss


def _model(path):
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    return get_peft_model(
        base,
        LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )


def _microbatch(model, tokenizer, row, collator, build, author_loss):
    inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in collator([{"problem": row["problem"], "solution": row["metadata"]["solution"]}]).items()
    }
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["student_prompts"],
            attention_mask=inputs["student_prompt_attention_mask"],
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.1,
            top_p=0.95,
            top_k=20,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    inputs = build(
        SimpleNamespace(processing_class=tokenizer), inputs, generated, generated.ne(tokenizer.pad_token_id).long()
    )
    response = generated[:, inputs["student_prompt_length"] :]
    with torch.no_grad(), model.disable_adapter():
        teacher = model(input_ids=inputs["teacher_input_ids"], attention_mask=inputs["teacher_attention_mask"]).logits
        teacher = teacher[:, inputs["teacher_prompt_length"] - 1 : -1].float()
    model.train()
    student = model(input_ids=inputs["student_input_ids"], attention_mask=inputs["student_attention_mask"]).logits
    student = student[:, inputs["student_prompt_length"] - 1 : -1].float()
    labels = inputs["labels"][:, inputs["student_prompt_length"] :]
    loss = author_loss(student, teacher, labels, 0.0, 1.1, token_clip=0.05)
    q = (teacher[0] / 1.1).log_softmax(-1).detach()
    mask = labels[0].ne(-100)
    candidate = opsd_per_token_loss(student[0], q, vocab_size=model.config.vocab_size, config=OPSDLossConfig())[
        mask
    ].mean()
    torch.testing.assert_close(loss, candidate, atol=2e-6, rtol=2e-5)
    a = torch.autograd.grad(loss, student, retain_graph=True)[0]
    b = torch.autograd.grad(candidate, student, retain_graph=True)[0]
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    metrics = dict(
        loss=loss.item(),
        response_length=response.size(1),
        loss_abs_error=abs(loss.item() - candidate.item()),
        logit_gradient_max_abs_error=(a - b).abs().max().item(),
    )
    del candidate, a, b, q
    (loss / 4).backward()
    return metrics, inputs, teacher[:, :8].detach().cpu(), response[0].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    collator = load_author_collator(args.reference_dir)(tokenizer, max_length=4096, reason_first=False)
    model = _model(args.model_dir)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-6, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()][:4]
    assert len(rows) == 4
    start = time.monotonic()
    metrics, tapes = [], []
    for row in rows:
        result, inputs, teacher, response = _microbatch(
            model,
            tokenizer,
            row,
            collator,
            load_author_input_builder(args.reference_dir),
            load_author_loss(args.reference_dir),
        )
        metrics.append(result)
        tapes.append({"problem_id": row["metadata"]["problem_id"], "response_ids": response})
        print(json.dumps(result), flush=True)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 0.1)
    assert torch.isfinite(norm) and norm > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        after = model(input_ids=inputs["teacher_input_ids"], attention_mask=inputs["teacher_attention_mask"]).logits
        after = after[:, inputs["teacher_prompt_length"] - 1 : -1][:, :8].float().cpu()
    torch.testing.assert_close(teacher, after, atol=0, rtol=0)
    model.save_pretrained(args.output / "adapter")
    (args.output / "response-tapes.json").write_text(json.dumps(tapes))
    result = dict(
        status="passed",
        optimizer_steps=1,
        microbatches=metrics,
        gradient_norm=norm.item(),
        teacher_bitwise_stable=True,
        elapsed_seconds=time.monotonic() - start,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        scope="HF full-checkpoint update against pinned author loss; not cross-framework update parity",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
