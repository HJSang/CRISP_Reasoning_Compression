"""Independent full-parameter HF replay of one short Megatron OPSD update.

The capture hook exports FP32 gradients in HF layout without modifying weights.
The replay uses the pinned author's loss on identical response tapes, FP32 Adam
masters, and the study optimizer. This is a diagnostic, never a training hook
for scientific runs. Keep the model-sized artifacts in a private output root.
"""

import argparse
import json
from pathlib import Path

import safetensors
import torch
from tests.opsd_reference import load_author_loss
from transformers import AutoModelForCausalLM


def capture_gradients(args, rollout_id, step_id, model, optimizer, scheduler):
    # The independent HF process does not need the Megatron runtime imports.
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
    from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
    from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher

    if rollout_id != 0 or step_id != 0 or args.lora_rank != 0 or args.num_rollout != 1:
        raise ValueError("Full gradient capture requires a single full-parameter replay update")
    iterator = get_hf_weight_iterator(
        args,
        model,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name="qwen3",
        quantization_config=None,
    )
    publisher = SnapshotPublisher(iterator)
    original_step = optimizer.step

    def capture_then_step(*positional, **keywords):
        parameters = dict(named_params_and_buffers(args, model, convert_to_global_name=False))
        if any(not p.requires_grad or not hasattr(p, "main_grad") for p in parameters.values()):
            raise ValueError("Full update replay requires a trainable gradient for every model parameter")
        gradients = {name: p.main_grad.detach() for name, p in parameters.items()}
        write_checkpoint_dir(
            Path(args.save).parent / "mapped-gradients",
            lambda directory: publisher.write_model(directory, weights=gradients, hf_checkpoint=args.hf_checkpoint),
            overwrite=False,
        )
        return original_step(*positional, **keywords)

    optimizer.step = capture_then_step


class TensorFiles:
    """Read one named tensor at a time; avoid concatenating billions of values."""

    def __init__(self, directory):
        self.directory = directory
        self.mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name):
        with safetensors.safe_open(self.directory / self.mapping[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)


def _sums(expected, actual):
    sums = torch.zeros(4, dtype=torch.float64)
    for start in range(0, expected.numel(), 1_000_000):
        x = expected.reshape(-1)[start : start + 1_000_000].double()
        y = actual.reshape(-1)[start : start + 1_000_000].double()
        sums += torch.stack(((x * y).sum(), x.square().sum(), y.square().sum(), (x - y).square().sum()))
    return sums


def _metrics(sums):
    dot, expected_sq, actual_sq, error_sq = sums.tolist()
    if expected_sq == 0 or actual_sq == 0:
        raise ValueError("A zero update cannot establish mapped update agreement")
    return dict(cosine=dot / (expected_sq * actual_sq) ** 0.5, relative_l2_error=(error_sq / expected_sq) ** 0.5)


def _replay(args):
    samples = torch.load(args.tape, map_location="cpu", weights_only=False)["samples"]
    if len(samples) != 4 or any(s["response_length"] > 64 for s in samples):
        raise ValueError("Use exactly four short, identical response tapes for the diagnostic")
    author_loss = load_author_loss(str(args.reference))
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .cuda()
        .eval()
    )
    parameters = list(model.named_parameters())
    masters = [torch.nn.Parameter(p.detach().float().clone()) for _, p in parameters]
    for master in masters:
        master.grad = torch.zeros_like(master)
    optimizer = torch.optim.Adam(masters, lr=5e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False)
    losses = []
    for sample in samples:
        tokens, length = sample["tokens"], sample["response_length"]
        prefix = sample["teacher_prompt_ids"]
        with torch.no_grad():
            teacher = model(torch.tensor([prefix + tokens[-length:]], device="cuda")).logits
            teacher = teacher[:, len(prefix) - 1 : -1].detach()
        student = model(torch.tensor([tokens], device="cuda")).logits[:, len(tokens) - length - 1 : -1]
        labels = torch.tensor([sample["loss_mask"]], device="cuda").long()
        labels = labels.masked_fill(labels == 0, -100)
        # The study and original trainer evaluate distribution math in FP32,
        # even though both model forwards and published weights use BF16.
        loss = author_loss(student.float(), teacher.float(), labels, 0.0, 1.0, token_clip=None)
        losses.append(float(loss.detach()))
        (loss / 4).backward()
        # Match Megatron's FP32 accumulation across samples; do not accumulate
        # four gradients into a BF16 leaf buffer before promoting precision.
        for (_, parameter), master in zip(parameters, masters, strict=True):
            if parameter.grad is None:
                raise ValueError("HF full-parameter replay found a missing gradient")
            master.grad.add_(parameter.grad.float())
            parameter.grad = None
        del loss, student, teacher
    return parameters, masters, optimizer, losses


def _compare(args, parameters, masters, optimizer, losses):
    before, after, gradients = (TensorFiles(path) for path in (args.model, args.after, args.gradients))
    keys = {name for name, _ in parameters}
    if any(keys != set(files.mapping) for files in (before, after, gradients)):
        raise ValueError("HF parameter names and exported checkpoint mappings differ")
    gradient_sums = torch.zeros(4, dtype=torch.float64)
    for (name, _), master in zip(parameters, masters, strict=True):
        gradient_sums += _sums(master.grad.cpu(), gradients.get(name))
    norm = float(torch.nn.utils.clip_grad_norm_(masters, 0.1))
    optimizer.step()
    update_sums = torch.zeros(4, dtype=torch.float64)
    maximum = 0.0
    for (name, parameter), master in zip(parameters, masters, strict=True):
        initial = before.get(name).float()
        expected = master.detach().to(parameter.dtype).cpu().float()
        actual = after.get(name).float()
        update_sums += _sums(expected - initial, actual - initial)
        maximum = max(maximum, float((expected - actual).abs().max()))
    update = _metrics(update_sums)
    passed = update["cosine"] >= 0.95 and update["relative_l2_error"] <= 0.35 and maximum <= 2e-5
    result = dict(
        status="passed" if passed else "failed",
        training_mode="full_parameter",
        examples=4,
        mean_loss=sum(losses) / 4,
        grad_norm=norm,
        mapped_gradients=_metrics(gradient_sums),
        mapped_update=update,
        max_parameter_error=maximum,
        tolerances=dict(
            minimum_update_cosine=0.95, maximum_update_relative_l2_error=0.35, maximum_parameter_error=2e-5
        ),
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise AssertionError("Full HF/Megatron update failed the declared BF16 tolerances")


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "after", "gradients", "tape", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _compare(args, *_replay(args))


if __name__ == "__main__":
    main()
