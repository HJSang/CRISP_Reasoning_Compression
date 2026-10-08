"""Ten-step tiny-Qwen FP32 replay against the pinned author loss, on one GPU.

Example from the Miles root:
  python tests/manual/opsd/check_hf_replay.py --reference-dir /path/to/OPSD --output /tmp/replay.json

Both branches use identical HF models and adapters. This isolates the OPSD loss
and reduction; it does not establish HF-to-Megatron adapter-update equivalence.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from miles.backends.training_utils.loss.hub.opsd import OPSDTarget, reference_loss
from miles.backends.training_utils.loss.hub.opsd_math import OPSDLossConfig
from tests.opsd_reference import load_author_loss


def _run(reference_dir):
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    author_loss = load_author_loss(reference_dir)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    base = Qwen3ForCausalLM(config).cuda().float()
    base.config._attn_implementation = "eager"
    reference = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    candidate = copy.deepcopy(reference)
    students = [reference, candidate]
    optimizers = [
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6, weight_decay=0)
        for model in students
    ]
    tokens = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    teacher_ids = torch.tensor([[1, 8, 9, 2, 4, 5]], device="cuda")
    labels = tokens[:, -2:]
    with torch.no_grad(), reference.disable_adapter():
        teacher = reference(teacher_ids).logits[:, 3:5].detach()
    q = (teacher[0] / 1.1).log_softmax(-1).cpu()
    batch = dict(
        total_lengths=[5],
        response_lengths=[2],
        unconcat_tokens=[tokens[0]],
        loss_masks=[torch.ones(2, device="cuda")],
        sample_indices=[0],
        opsd_rollout_ids=[0],
        opsd_targets=[OPSDTarget(0, 0, (4, 5), 1.1, 128, 0, q)],
    )
    errors = {"loss": 0.0, "gradient": 0.0, "parameter": 0.0, "optimizer_moment": 0.0}
    for _ in range(10):
        for optimizer in optimizers:
            optimizer.zero_grad()
        ref_loss = author_loss(reference(tokens).logits[:, 2:4], teacher, labels, 0.0, 1.1, token_clip=0.05)
        got_loss, _ = reference_loss(candidate(tokens).logits, batch, config=OPSDLossConfig(), vocab_size=128)
        errors["loss"] = max(errors["loss"], abs((ref_loss - got_loss).item()))
        torch.testing.assert_close(ref_loss, got_loss, atol=2e-6, rtol=2e-5)
        ref_loss.backward()
        got_loss.backward()
        for a, b in zip(reference.parameters(), candidate.parameters(), strict=True):
            if a.requires_grad:
                errors["gradient"] = max(errors["gradient"], (a.grad - b.grad).abs().max().item())
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
            else:
                assert a.grad is None and b.grad is None
        for model, optimizer in zip(students, optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1)
            optimizer.step()
        for a, b in zip(reference.parameters(), candidate.parameters(), strict=True):
            errors["parameter"] = max(errors["parameter"], (a - b).abs().max().item())
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            if a.requires_grad:
                for key in ("exp_avg", "exp_avg_sq"):
                    x, y = optimizers[0].state[a][key], optimizers[1].state[b][key]
                    errors["optimizer_moment"] = max(errors["optimizer_moment"], (x - y).abs().max().item())
                    torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
        with torch.no_grad(), reference.disable_adapter():
            torch.testing.assert_close(reference(teacher_ids).logits[:, 3:5], teacher, atol=0, rtol=0)
    return dict(
        status="passed", optimizer_steps=10, dtype="float32", max_abs_errors=errors, frozen_teacher_bitwise_stable=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _run(args.reference_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
