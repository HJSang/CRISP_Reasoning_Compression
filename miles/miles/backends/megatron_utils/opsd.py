"""Frozen-teacher prepass in Miles' existing model slot, before student autograd."""

from functools import partial

import torch

from miles.backends.megatron_utils.lora.utils import _is_adapter_param_name
from miles.backends.megatron_utils.model import forward_only
from miles.backends.training_utils.data.opsd import teacher_batch
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.loss.hub.opsd import OPSDTarget, response_logits
from miles.backends.training_utils.loss.hub.opsd_math import vocab_log_softmax
from miles.backends.training_utils.parallel import get_parallel_state


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


def verify_teacher_base(backuper):
    """The initial self-distillation recipe uses the actor's identical frozen base."""
    actor, teacher = backuper.get("actor"), backuper.get("teacher")
    if actor.keys() != teacher.keys():
        raise ValueError("OPSD actor and teacher snapshots have different parameter sets")
    for name in actor:
        if not _is_adapter_param_name(name) and not torch.equal(actor[name], teacher[name]):
            raise ValueError(
                "OPSD teacher base differs from the frozen actor base; checkpoint ablations require a separate recipe"
            )


def score_teacher(actor, rollout_data, num_microbatches, *, rollout_id):
    args = actor.args
    tp = get_parallel_state().tp
    if sum(num_microbatches) * args.micro_batch_size != len(rollout_data["tokens"]):
        raise ValueError("OPSD requires the fixed microbatch schedule to cover the entire local rollout")
    width = args.padded_vocab_size // tp.size
    real_width = min(width, max(0, args.vocab_size - tp.rank * width))
    data = teacher_batch(
        rollout_data,
        seq_length=args.seq_length,
        local_vocab_width=real_width,
        cache_bytes=int(args.opsd_target_cache_gib * 1024**3),
        microbatch_bytes=int(args.opsd_target_microbatch_gib * 1024**3),
        micro_batch_size=args.micro_batch_size,
    )
    iterator = [DataIterator(data, micro_batch_size=args.micro_batch_size)]
    prior_modes = [module.training for module in actor.model]
    # All target computation and weight restoration precede the first student
    # graph. The finally also restores state after a failed scoring callback.
    try:
        actor._switch_model("teacher")
        targets = forward_only(
            partial(_collect_targets, rollout_id=rollout_id),
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
    if [target.sample_index for target in targets] != list(data["sample_indices"]):
        raise ValueError("OPSD teacher prepass changed sample ordering")
    rollout_data["opsd_targets"] = targets
    rollout_data["opsd_rollout_ids"] = [rollout_id] * len(targets)


@torch.no_grad()
def _collect_targets(
    logits, *, args, unconcat_tokens, total_lengths, response_lengths, sample_indices, rollout_id, **_
):
    tp = get_parallel_state().tp
    if logits.size(-1) * tp.size != args.padded_vocab_size:
        raise ValueError("OPSD requires vocabulary-sharded model logits matching the padded vocabulary")
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
