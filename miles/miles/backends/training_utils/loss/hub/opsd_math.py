"""Full-vocabulary OPSD divergence, with differentiable vocabulary parallelism.

The pinned reference is siyan-zhao/OPSD at ae7d2519. In particular, clipping
applies to each vocabulary contribution, before the vocabulary reduction.
Teacher inputs are normalized log-probabilities, not sampled-token scores.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class OPSDLossConfig:
    beta: float = 0.0
    temperature: float = 1.1
    token_clip: float | None = 0.05

    def __post_init__(self):
        if not 0 <= self.beta <= 1:
            raise ValueError("OPSD beta must be in [0, 1]")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("OPSD loss temperature must be finite and positive")
        if self.token_clip is not None and (not math.isfinite(self.token_clip) or self.token_clip < 0):
            raise ValueError("OPSD token clip must be finite and nonnegative, or None")


def _group_size(group: dist.ProcessGroup | None) -> int:
    # None explicitly means an unsharded vocabulary, not the default world group.
    return 1 if group is None else dist.get_world_size(group)


class _VocabLogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, group):
        maximum = (
            logits.amax(-1, keepdim=True) if logits.size(-1) else logits.new_full((*logits.shape[:-1], 1), -math.inf)
        )
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        shifted = logits - maximum
        denominator = shifted.exp().sum(-1, keepdim=True)
        dist.all_reduce(denominator, group=group)
        result = shifted - denominator.log()
        ctx.group = group
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (log_probs,) = ctx.saved_tensors
        total_grad = grad_output.sum(-1, keepdim=True)
        dist.all_reduce(total_grad, group=ctx.group)
        return grad_output - log_probs.exp() * total_grad, None


class _SumVocab(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        result = value.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # The scalar loss is replicated across TP. Reducing backward again would
        # multiply its gradient by TP size (unlike sharding the output loss).
        return grad_output, None


def vocab_log_softmax(
    logits: torch.Tensor,
    *,
    vocab_size: int,
    temperature: float,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Normalize [response positions, local padded vocabulary] over real IDs.

    Returned columns exclude the local vocabulary padding. FP64 is retained for
    oracle tests; FP16/BF16 inputs use FP32 for the distribution calculation.
    """
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("OPSD logits must be a floating [positions, local_vocab] tensor")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("OPSD loss temperature must be finite and positive")
    size = _group_size(group)
    width = logits.size(-1)
    if not 0 < vocab_size <= width * size:
        raise ValueError("Real vocabulary must fit the padded vocabulary shards")
    rank = 0 if size == 1 else dist.get_rank(group)
    valid_width = min(width, max(0, vocab_size - rank * width))
    values = logits[:, :valid_width]
    if values.dtype != torch.float64:
        values = values.float()
    values = values / temperature
    return F.log_softmax(values, dim=-1) if size == 1 else _VocabLogSoftmax.apply(values, group)


def opsd_per_token_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    vocab_size: int,
    config: OPSDLossConfig,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Return one divergence per response position, with no sequence reduction.

    Teacher log probabilities must already use config.temperature and the same
    global vocabulary/shard. This boundary always detaches teacher targets.
    """
    student = vocab_log_softmax(student_logits, vocab_size=vocab_size, temperature=config.temperature, group=group)
    teacher = teacher_log_probs.detach().to(device=student.device, dtype=student.dtype)
    if teacher.shape != student.shape:
        raise ValueError("Teacher target shape must match response positions and real local vocabulary")
    if config.beta == 0:
        terms = F.kl_div(student, teacher, reduction="none", log_target=True)
    elif config.beta == 1:
        terms = F.kl_div(teacher, student, reduction="none", log_target=True)
    else:
        beta = student.new_tensor(config.beta)
        mixture = torch.logsumexp(torch.stack((student + torch.log1p(-beta), teacher + beta.log())), dim=0)
        terms = beta * F.kl_div(mixture, teacher, reduction="none", log_target=True)
        terms = terms + (1 - beta) * F.kl_div(mixture, student, reduction="none", log_target=True)
    if config.token_clip is not None:
        terms = terms.clamp(max=config.token_clip)
    local = terms.sum(-1)
    return local if _group_size(group) == 1 else _SumVocab.apply(local, group)
