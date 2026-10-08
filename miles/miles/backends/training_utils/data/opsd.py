"""Prepare privileged sequences and bound dense-target storage before scoring."""

import torch


def teacher_batch(rollout_data, *, seq_length, local_vocab_width, cache_bytes, microbatch_bytes, micro_batch_size):
    prefixes = rollout_data.get("teacher_prompt_ids")
    if prefixes is None or len(prefixes) != len(rollout_data["tokens"]):
        raise ValueError("OPSD requires a separately templated teacher prefix for every sample")
    indices = rollout_data["sample_indices"]
    if any(index is None for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("OPSD requires unique stable sample indices within a rollout")
    if len(indices) % micro_batch_size:
        raise ValueError("OPSD local rollout must contain whole fixed microbatches")
    target_bytes = [response * local_vocab_width * 4 for response in rollout_data["response_lengths"]]
    if sum(target_bytes) > cache_bytes:
        raise ValueError("OPSD teacher targets exceed --opsd-target-cache-gib")
    for start in range(0, len(indices), micro_batch_size):
        if sum(target_bytes[start : start + micro_batch_size]) > microbatch_bytes:
            raise ValueError("OPSD teacher targets exceed --opsd-target-microbatch-gib")
        # The reference mean is undefined when all local labels are masked.
        if (
            sum(
                int(torch.as_tensor(mask).sum())
                for mask in rollout_data["loss_masks"][start : start + micro_batch_size]
            )
            == 0
        ):
            raise ValueError("OPSD reference reduction requires valid tokens in every microbatch")
    teacher_tokens = []
    for tokens, prefix, response, total in zip(
        rollout_data["tokens"], prefixes, rollout_data["response_lengths"], rollout_data["total_lengths"], strict=True
    ):
        if len(tokens) != total or not 0 <= response < total or not len(prefix):
            raise ValueError("Invalid OPSD prefix/response boundaries")
        prefix = torch.as_tensor(prefix, dtype=tokens.dtype, device=tokens.device)
        teacher_tokens.append(torch.cat((prefix, tokens[total - response :])))
        if max(total, len(teacher_tokens[-1])) > seq_length:
            raise ValueError("OPSD student or teacher sequence exceeds --seq-length; no truncation is permitted")
    return {
        "tokens": teacher_tokens,
        "total_lengths": [len(tokens) for tokens in teacher_tokens],
        "response_lengths": rollout_data["response_lengths"],
        "loss_masks": rollout_data["loss_masks"],
        "sample_indices": indices,
    }
