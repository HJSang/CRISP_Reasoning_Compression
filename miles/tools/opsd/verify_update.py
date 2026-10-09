"""Capture a Megatron replay's initial adapter and independently replay its HF update.

The capture hook is for a one-update correctness replay only. Both runtimes must
start with the exported adapter and consume identical token IDs and PI prefixes.
The check reports mapped adapter-update direction and relative error separately
from absolute parameter error; small learning rates cannot hide a wrong update.
"""

import argparse
import json
import types
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from miles.backends.training_utils.loss.hub.opsd import OPSDTarget, reference_loss
from miles.backends.training_utils.loss.hub.opsd_math import OPSDLossConfig


def capture_initial_adapter(args, rollout_id, step_id, model, optimizer, scheduler):
    # Optional Megatron imports: the HF replay runs in an independent single-GPU process.
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
    from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
    from miles.utils.lora.utils import build_lora_config

    if rollout_id != 0 or step_id != 0:
        raise ValueError("Mapped update capture is restricted to a single replay update")
    iterator = get_hf_weight_iterator(
        args,
        model,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name="qwen3",
        quantization_config=None,
    )
    publisher = SnapshotPublisher(iterator, build_lora_config(args, target_modules=args.lora_adapter_targets))
    publisher.publish_adapter(None, str(Path(args.save).parent / "initial-adapter"))
    _capture_forward(model, Path(args.save).parent)
    original_step = optimizer.step

    def capture_gradients_then_step(*step_args, **step_kwargs):
        parameters = [parameter for chunk in model for parameter in chunk.parameters() if parameter.requires_grad]
        original = [parameter.detach().clone() for parameter in parameters]
        try:
            with torch.no_grad():
                for parameter in parameters:
                    parameter.copy_(parameter.main_grad)
            publisher.publish_adapter(None, str(Path(args.save).parent / "initial-gradients"))
        finally:
            with torch.no_grad():
                for parameter, saved in zip(parameters, original, strict=True):
                    parameter.copy_(saved)
        return original_step(*step_args, **step_kwargs)

    optimizer.step = capture_gradients_then_step


def _capture_forward(model, directory):
    """Short replay only: capture the first packed student forward for bisection."""
    captured = {}
    probes = {}
    for name, parameter in model[0].named_parameters():
        if name.endswith("embedding.word_embeddings.weight"):
            probes[name] = parameter[:128].detach().cpu()
        elif name.endswith("decoder.final_layernorm.weight") or (
            "layers.0." in name and ("layernorm.weight" in name or "layer_norm_weight" in name)
        ):
            probes[name] = parameter.detach().cpu()
    torch.save(probes, directory / f"base-probes-rank{dist.get_rank()}.pt")

    def record(name):
        def hook(module, inputs, output):
            if name not in captured:
                tensor = output[0] if isinstance(output, tuple) else output
                captured[name] = tensor.detach().cpu()

        return hook

    def record_input(module, inputs, kwargs):
        if "input_ids" not in captured:
            captured["input_ids"] = (kwargs["input_ids"] if "input_ids" in kwargs else inputs[0]).detach().cpu()

    model[0].register_forward_pre_hook(record_input, with_kwargs=True)
    for name, module in model[0].named_modules():
        for suffix in [
            "embedding",
            "decoder.layers.0",
            "decoder.layers.17",
            "decoder.layers.35",
            "decoder.final_layernorm",
            "decoder.layers.0.self_attention.linear_qkv",
            "decoder.layers.0.self_attention.q_layernorm",
            "decoder.layers.0.self_attention.k_layernorm",
            "decoder.layers.0.self_attention.core_attention",
            "decoder.layers.0.self_attention.linear_proj",
            "decoder.layers.0.mlp.linear_fc1",
            "decoder.layers.0.mlp.linear_fc2",
        ]:
            if name.endswith(suffix):
                module.register_forward_hook(record(suffix))

    def finish(module, inputs, output):
        destination = directory / f"initial-forward-rank{dist.get_rank()}.pt"
        if not destination.exists():
            torch.save(captured, destination)

    model[0].register_forward_hook(finish)


def _run(args):
    tape = torch.load(args.tape, map_location="cpu", weights_only=False)
    samples = tape["samples"]
    if len(samples) != 4:
        raise ValueError("Mapped study update requires the four-sample effective batch")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).cuda()
    model = get_peft_model(base, LoraConfig.from_pretrained(args.initial)).to(dtype=torch.bfloat16).eval()
    if args.match_norm_arithmetic:
        for module in model.modules():
            if type(module).__name__ == "Qwen3RMSNorm":
                module.forward = types.MethodType(_fused_rmsnorm, module)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial = safetensors.torch.load_file(str(args.initial / "adapter_model.safetensors"))
    keys = {name.removeprefix("base_model.model.").replace(".default.", ".") for name, _ in parameters}
    if keys != initial.keys():
        raise ValueError("HF trainable adapter names do not match the exported checkpoint")
    with torch.no_grad():
        for name, parameter in parameters:
            key = name.removeprefix("base_model.model.").replace(".default.", ".")
            parameter.copy_(initial[key])
            torch.testing.assert_close(parameter.cpu(), initial[key], atol=0, rtol=0)
    # Megatron accumulates/updates FP32 masters while forward parameters are BF16.
    masters = [torch.nn.Parameter(parameter.detach().float().clone()) for _, parameter in parameters]
    optimizer = torch.optim.Adam(masters, lr=5e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    losses = []
    for index, sample in enumerate(samples):
        tokens = torch.tensor(sample["tokens"], device="cuda")
        length = sample["response_length"]
        prefix = sample["teacher_prompt_ids"]
        with torch.no_grad(), model.disable_adapter():
            teacher_ids = torch.tensor([prefix + sample["tokens"][-length:]], device="cuda")
            q = model(teacher_ids).logits[0, len(prefix) - 1 : -1].float().log_softmax(-1).cpu()
        logits = model(tokens[None]).logits
        batch = dict(
            total_lengths=[len(tokens)],
            response_lengths=[length],
            unconcat_tokens=[tokens],
            loss_masks=[torch.tensor(sample["loss_mask"], device="cuda")],
            sample_indices=[index],
            opsd_rollout_ids=[0],
            opsd_targets=[OPSDTarget(index, 0, tuple(tokens[-length:].tolist()), 1.0, q.size(-1), 0, q)],
        )
        loss, _ = reference_loss(
            logits,
            batch,
            config=OPSDLossConfig(temperature=1.0, token_clip=None),
            vocab_size=q.size(-1),
            reduction="sample_mean",
        )
        losses.append(float(loss.detach()))
        (loss / len(samples)).backward()
    for master, (_, parameter) in zip(masters, parameters, strict=True):
        master.grad = parameter.grad.float()
    gradient_metrics = None
    if args.gradients is not None:
        mapped = safetensors.torch.load_file(str(args.gradients / "adapter_model.safetensors"))
        expected_grad = torch.cat([master.grad.detach().cpu().flatten() for master in masters]).double()
        actual_grad = torch.cat(
            [
                mapped[name.removeprefix("base_model.model.").replace(".default.", ".")].float().flatten()
                for name, _ in parameters
            ]
        ).double()
        gradient_metrics = dict(
            cosine=float(torch.nn.functional.cosine_similarity(expected_grad, actual_grad, dim=0)),
            relative_l2_error=float((actual_grad - expected_grad).norm() / expected_grad.norm()),
        )
    grad_norm = float(torch.nn.utils.clip_grad_norm_(masters, 0.1))
    optimizer.step()
    actual = safetensors.torch.load_file(str(args.after / "adapter_model.safetensors"))
    expected_deltas, actual_deltas, parameter_errors = [], [], []
    for (name, parameter), master in zip(parameters, masters, strict=True):
        key = name.removeprefix("base_model.model.").replace(".default.", ".")
        expected = master.detach().to(parameter.dtype).cpu().float()
        before = initial[key].float()
        got = actual[key].float()
        expected_deltas.append((expected - before).flatten())
        actual_deltas.append((got - before).flatten())
        parameter_errors.append(float((expected - got).abs().max()))
    # FP64 reductions avoid inaccurate norms over millions of tiny BF16 deltas.
    expected, got = torch.cat(expected_deltas).double(), torch.cat(actual_deltas).double()
    cosine = float(torch.nn.functional.cosine_similarity(expected, got, dim=0))
    relative = float((got - expected).norm() / expected.norm().clamp_min(1e-12))
    # Predeclared BF16 cross-backend tolerances, not exact optimizer equivalence.
    passed = cosine >= 0.95 and relative <= 0.35 and max(parameter_errors) <= 2e-5
    result = dict(
        status="passed" if passed else "failed",
        mean_loss=sum(losses) / len(losses),
        grad_norm=grad_norm,
        update_cosine=cosine,
        update_relative_l2_error=relative,
        max_parameter_error=max(parameter_errors),
        tolerances={
            "minimum_update_cosine": 0.95,
            "maximum_update_relative_l2_error": 0.35,
            "maximum_parameter_error": 2e-5,
        },
        examples=4,
        match_norm_arithmetic=args.match_norm_arithmetic,
        mapped_gradients=gradient_metrics,
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise AssertionError("Mapped HF/Megatron update does not satisfy the declared BF16 tolerances")


def _fused_rmsnorm(module, inputs):
    # Controlled precision diagnostic: TE multiplies the RMSNorm weight before
    # the BF16 cast; vanilla HF casts the normalized activation before multiplying.
    values = inputs.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + module.variance_epsilon)
    return (normalized * module.weight.float()).to(inputs.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--match-norm-arithmetic", action="store_true")
    parser.add_argument("--gradients", type=Path)
    for name in ["initial", "after", "tape", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
