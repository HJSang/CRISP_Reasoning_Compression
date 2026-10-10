"""Detached teacher prepass in Miles' existing model slot, before student autograd."""

import math
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora.utils import _is_adapter_param_name
from miles.backends.training_utils.data.opsd import teacher_batch
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.loss.hub.opsd import OPSDTarget, response_logits
from miles.backends.training_utils.loss.hub.opsd_math import vocab_log_softmax
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.opsd_checkpoint import checkpoint_digest
from miles.utils.opsd_ema import OPSDEMA


def initialize_ema(actor):
    decay = getattr(actor.args, "opsd_teacher_ema_decay", None)
    actor.opsd_ema = None
    if decay is None:
        return
    actor.weights_backuper.backup("ema_teacher")
    actor.opsd_ema = OPSDEMA(actor.weights_backuper.get("teacher"), decay=decay)
    if actor.args.opsd_ema_source is not None:
        actor.opsd_ema.load(
            actor.args.opsd_ema_source / f"rank-{dist.get_rank():05d}.pt",
            student_sha256=checkpoint_digest(Path(actor.args.load)),
        )
    actor.opsd_ema.publish(actor.weights_backuper.get("ema_teacher"))


def save_ema_source(actor, rollout_id):
    """Keep only the final paired teacher, under the corresponding HF student source."""
    if actor.opsd_ema is None or not actor.args.opsd_retain_final_eval_snapshot:
        return
    if rollout_id + 1 != actor.args.num_rollout:
        return
    # HF export's final marker is written by rank zero after its last collective.
    # Other ranks must wait before hashing that completed student source.
    dist.barrier(group=get_gloo_group())
    checkpoint = Path(actor.args.save_hf.format(rollout_id=rollout_id))
    actor.opsd_ema.save(
        checkpoint / "ema-teacher" / f"rank-{dist.get_rank():05d}.pt",
        student_sha256=checkpoint_digest(checkpoint),
    )
    dist.barrier(group=get_gloo_group())


@torch.no_grad()
def zero_teacher_adapters(model):
    """A frozen base checkpoint must not retain the student's live adapters.

    Only called while initializing the separate teacher snapshot, after the actor
    has been backed up. Both factors are zeroed: the canonical additive branch then
    contributes exactly zero. The actor snapshot restores its original parameters.
    """
    count = 0
    for chunk in model:
        for name, param in chunk.named_parameters():
            if _is_adapter_param_name(name):
                param.zero_()
                count += 1
            elif param.requires_grad:
                raise ValueError("OPSD frozen-base mode requires every non-adapter parameter to be frozen")
    if count == 0:
        raise ValueError("OPSD could not identify the canonical LoRA parameters")


def verify_teacher_base(backuper, *, full_parameter=False):
    """LoRA shares a frozen base; full-parameter branches may start from a warmed actor."""
    actor, teacher = backuper.get("actor"), backuper.get("teacher")
    if actor.keys() != teacher.keys():
        raise ValueError("OPSD actor and teacher snapshots have different parameter sets")
    for name in actor:
        if actor[name].shape != teacher[name].shape or actor[name].dtype != teacher[name].dtype:
            raise ValueError("OPSD actor and teacher snapshots have incompatible tensors")
        if full_parameter:
            if not torch.isfinite(teacher[name]).all():
                raise ValueError("OPSD frozen teacher contains nonfinite parameters")
            continue
        if not _is_adapter_param_name(name) and not torch.equal(actor[name], teacher[name]):
            raise ValueError(
                "OPSD teacher base differs from the frozen actor base; checkpoint ablations require a separate recipe"
            )


def score_teacher(actor, rollout_data, num_microbatches, *, rollout_id):
    # GPU runtime dependencies stay optional for CPU snapshot-contract checks.
    from megatron.core.utils import get_attr_wrapped_model

    args = actor.args
    tp = get_parallel_state().tp
    if sum(num_microbatches) * args.micro_batch_size != len(rollout_data["tokens"]):
        raise ValueError("OPSD requires the fixed microbatch schedule to cover the entire local rollout")
    # Bridge can disable padding even when Megatron's CLI computes a larger
    # padded_vocab_size. Budget and validate against the constructed output head.
    if len(actor.model) != 1 or not get_attr_wrapped_model(actor.model[0], "parallel_output"):
        raise ValueError("OPSD requires one model chunk with vocabulary-parallel output")
    model_vocab_size = get_attr_wrapped_model(actor.model[0], "vocab_size")
    if model_vocab_size < args.vocab_size or model_vocab_size % tp.size:
        raise ValueError("OPSD model vocabulary must cover the HF vocabulary and divide evenly across TP")
    width = model_vocab_size // tp.size
    real_width = min(width, max(0, args.vocab_size - tp.rank * width))
    if getattr(args, "opsd_target", "frozen") == "transported":
        staging_bytes = 4 * sum(rollout_data["response_lengths"]) * real_width * 4
        if staging_bytes > args.opsd_aggregate_staging_gib * 1024**3:
            raise ValueError("Transported OPSD targets exceed the aggregate CPU staging budget")
    data = teacher_batch(
        rollout_data,
        seq_length=args.seq_length,
        local_vocab_width=real_width,
        cache_bytes=int(args.opsd_target_cache_gib * 1024**3),
        microbatch_bytes=int(args.opsd_target_microbatch_gib * 1024**3),
        micro_batch_size=args.micro_batch_size,
    )
    target_mode = getattr(args, "opsd_target", "frozen")
    teacher_tag = "ema_teacher" if getattr(args, "opsd_teacher_ema_decay", None) is not None else "teacher"
    targets = _score(
        actor,
        data,
        num_microbatches,
        rollout_id=rollout_id,
        width=width,
        model_tag="actor" if target_mode == "current" else teacher_tag,
    )
    if actor.opsd_ema is not None and actor.opsd_ema.updates == 0:
        # An EMA initialized at the original checkpoint must produce the exact
        # frozen-teacher targets on the same response tape and privileged prefix.
        original_targets = _score(actor, data, num_microbatches, rollout_id=rollout_id, width=width, model_tag="teacher")
        for averaged, original in zip(targets, original_targets, strict=True):
            if _target_identity(averaged) != _target_identity(original):
                raise ValueError("Initial EMA target identity differs from the original teacher")
            torch.testing.assert_close(averaged.log_probs, original.log_probs, atol=0, rtol=0)
    if target_mode == "transported":
        # No-PI passes use the exact student sequence, never an empty context wrapper.
        unprivileged = {key: rollout_data[key] for key in data}
        reference = _score(
            actor, unprivileged, num_microbatches, rollout_id=rollout_id, width=width, model_tag="teacher"
        )
        student = _score(actor, unprivileged, num_microbatches, rollout_id=rollout_id, width=width, model_tag="actor")
        targets = _transport_targets(targets, reference, student, device=rollout_data["tokens"][0].device)
    if [target.sample_index for target in targets] != list(data["sample_indices"]):
        raise ValueError("OPSD teacher prepass changed sample ordering")
    rollout_data["opsd_targets"] = targets
    rollout_data["opsd_rollout_ids"] = [rollout_id] * len(targets)


def _score(actor, data, num_microbatches, *, rollout_id, width, model_tag):
    from miles.backends.megatron_utils.model import forward_only

    args = actor.args
    iterator = [DataIterator(data, micro_batch_size=args.micro_batch_size)]
    prior_modes = [module.training for module in actor.model]
    # All target computation and weight restoration precede the first student
    # graph. The finally also restores state after a failed scoring callback.
    try:
        actor._switch_model(model_tag)
        targets = forward_only(
            partial(_collect_targets, rollout_id=rollout_id, local_vocab_width=width),
            args=args,
            model=actor.model,
            data_iterator=iterator,
            num_microbatches=num_microbatches,
            rollout_id=rollout_id,
            store_prefix="",
            fp32_output=False,
        )["opsd_targets"]
    finally:
        actor._switch_model("actor")
        for module, training in zip(actor.model, prior_modes, strict=True):
            module.train(training)
    return targets


@torch.no_grad()
def _transport_targets(privileged, reference, student, *, device):
    """Normalize p*q/r over all real vocabulary shards with bounded GPU staging."""
    tp = get_parallel_state().tp
    result = []
    for q, r, p in zip(privileged, reference, student, strict=True):
        if not _target_identity(q) == _target_identity(r) == _target_identity(p):
            raise ValueError("Transported OPSD target alignment mismatch")
        if not q.log_probs.shape == r.log_probs.shape == p.log_probs.shape:
            raise ValueError("Transported OPSD target shape mismatch")
        output = torch.empty_like(q.log_probs)
        for start in range(0, len(output), 128):
            section = slice(start, start + 128)
            values = (
                p.log_probs[section].to(device) + q.log_probs[section].to(device) - r.log_probs[section].to(device)
            )
            maximum = (
                values.amax(-1, keepdim=True) if values.size(-1) else values.new_full((len(values), 1), -math.inf)
            )
            if tp.size > 1:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp.group)
            values -= maximum
            denominator = values.exp().sum(-1, keepdim=True)
            if tp.size > 1:
                dist.all_reduce(denominator, group=tp.group)
            output[section] = (values - denominator.log()).cpu()
        result.append(replace(q, log_probs=output))
    return result


def _target_identity(target):
    return (
        target.sample_index,
        target.rollout_id,
        target.response_ids,
        target.temperature,
        target.vocab_size,
        target.vocab_start,
    )


@torch.no_grad()
def _collect_targets(
    logits,
    *,
    args,
    unconcat_tokens,
    total_lengths,
    response_lengths,
    sample_indices,
    rollout_id,
    local_vocab_width,
    **_,
):
    tp = get_parallel_state().tp
    if logits.size(-1) != local_vocab_width:
        raise ValueError("OPSD logits must match the constructed model's vocabulary shard")
    targets = []
    for block, tokens, response, index in zip(
        response_logits(logits, total_lengths, response_lengths),
        unconcat_tokens,
        response_lengths,
        sample_indices,
        strict=True,
    ):
        log_probs = vocab_log_softmax(
            block,
            vocab_size=args.vocab_size,
            temperature=args.opsd_temperature,
            group=tp.group if tp.size > 1 else None,
        )
        # A synchronous CPU copy bounds residency even while the pipeline engine
        # retains previous callback results. No dense targets enter the object store.
        targets.append(
            OPSDTarget(
                sample_index=int(index),
                rollout_id=rollout_id,
                response_ids=tuple(tokens[-response:].tolist()) if response else (),
                temperature=args.opsd_temperature,
                vocab_size=args.vocab_size,
                vocab_start=tp.rank * logits.size(-1),
                log_probs=log_probs.cpu(),
            )
        )
    return {"opsd_targets": targets}
