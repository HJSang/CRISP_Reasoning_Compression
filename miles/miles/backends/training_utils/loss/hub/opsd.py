"""Worker-local dense targets and reference OPSD reduction (PP1, CP1, THD)."""

from dataclasses import dataclass

import torch

from miles.backends.training_utils.loss.hub.opsd_math import OPSDLossConfig, opsd_per_token_loss


@dataclass(frozen=True)
class OPSDTarget:
    sample_index: int
    rollout_id: int
    response_ids: tuple[int, ...]
    temperature: float
    vocab_size: int
    vocab_start: int
    log_probs: torch.Tensor  # Detached CPU [response positions, real local vocab].


def response_logits(logits, total_lengths, response_lengths):
    """Yield prediction rows for response IDs, never across a packed boundary."""
    if logits.ndim != 3 or logits.size(0) != 1:
        raise ValueError("OPSD expects THD logits shaped [1, packed_tokens, local_vocab]")
    offset = 0
    for total, response in zip(total_lengths, response_lengths, strict=True):
        if not 0 <= response < total:
            raise ValueError("OPSD requires a nonempty prefix and a valid response length")
        end = offset + total
        if end > logits.size(1):
            raise ValueError("OPSD packed lengths exceed the model output")
        yield logits[0, end - response - 1 : end - 1]
        offset = end


def reference_loss(logits, batch, *, config, vocab_size, vocab_start=0, group=None):
    """Mean over valid tokens in this microbatch, matching the author function.

    Megatron subsequently averages microbatches and DP ranks. Do not apply Miles'
    usual sum-of-sequence-means scaling or call this a global valid-token mean.
    """
    chunks = response_logits(logits, batch["total_lengths"], batch["response_lengths"])
    rows = zip(
        chunks,
        batch["opsd_targets"],
        batch["sample_indices"],
        batch["unconcat_tokens"],
        batch["loss_masks"],
        batch["opsd_rollout_ids"],
        strict=True,
    )
    losses, masks = [], []
    for student, target, sample_index, tokens, mask, rollout_id in rows:
        response = student.size(0)
        actual_ids = tuple(tokens[-response:].tolist()) if response else ()
        if (target.sample_index, target.rollout_id, target.response_ids) != (sample_index, rollout_id, actual_ids):
            raise ValueError("OPSD target sample/response alignment mismatch")
        if (target.temperature, target.vocab_size, target.vocab_start) != (config.temperature, vocab_size, vocab_start):
            raise ValueError("OPSD teacher/student distribution contract mismatch")
        if target.log_probs.device.type != "cpu" or target.log_probs.requires_grad:
            raise ValueError("OPSD cached targets must be detached CPU tensors")
        mask = torch.as_tensor(mask, device=logits.device)
        if mask.shape != (response,) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("OPSD requires one binary loss mask per response token")
        losses.append(opsd_per_token_loss(student, target.log_probs, vocab_size=vocab_size, config=config, group=group))
        masks.append(mask.bool())
    mask = torch.cat(masks)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("OPSD reference reduction is undefined for a microbatch with zero valid tokens")
    return torch.cat(losses)[mask].sum() / count, count


def opsd_loss_function(args, batch, logits):
    # Imported at the framework boundary; the numerical contract remains CPU-testable.
    from miles.backends.training_utils.parallel import get_parallel_state

    tp = get_parallel_state().tp
    config = OPSDLossConfig(
        beta=args.opsd_beta, temperature=args.opsd_temperature, token_clip=args.opsd_token_clip or None
    )
    loss, count = reference_loss(
        logits,
        batch,
        config=config,
        vocab_size=args.vocab_size,
        vocab_start=tp.rank * logits.size(-1),
        group=tp.group if tp.size > 1 else None,
    )
    # Logging averages microbatches, just as the reference loss does; token count
    # is a diagnostic, not its hidden denominator.
    values = torch.stack((loss.detach().new_tensor(1), loss.detach(), count.to(loss.dtype)))
    return (
        loss,
        torch.ones((), dtype=torch.int64, device=logits.device),
        {"keys": ["opsd_loss", "opsd_valid_tokens"], "values": values},
    )
