"""Pure scheduling rules for the bounded independent PI warm-up replication."""

import math
from dataclasses import dataclass
from pathlib import Path


SEEDS = (29, 43, 71)


@dataclass(frozen=True)
class Job:
    run_id: str
    seed: int
    context: str
    dataset: str
    updates: int
    snapshot_interval: int
    retain_final_snapshot: bool
    initial_checkpoint: str
    expected_evaluations: int


def paired_jobs(seed: int, model: Path, runs: Path) -> tuple[Job, Job, Job]:
    """Each pair shares exactly one independently generated, seven-update source."""
    if seed not in SEEDS:
        raise ValueError("Seed is outside the approved three-replicate stage")
    warm_id = f"replication-s{seed}-warmup"
    source = runs / warm_id / "eval-snapshots" / "step_6"
    warm = Job(warm_id, seed, "worked", f"warmup-{seed}.jsonl", 7, 7, True, str(model), 1)
    branches = tuple(
        Job(f"replication-s{seed}-{method}", seed, context, f"adaptation-{seed}.jsonl", 4, 1, False, str(source), 4)
        for method, context in (("opd", "none"), ("pi", "worked"))
    )
    return (warm, *branches)


def admit_remaining(*, used_gpu_hours: float, pair_gpu_hours: float, remaining_seconds: float,
                    warmup_seconds: float, branch_seconds: float) -> dict:
    """Reserve both remaining pairs, every eval drain, and cleanup before admission.

    Warm-ups run concurrently, followed by four concurrent branches. A 25% timing
    margin and five minutes for transfers/cleanup avoid spending the final unit of
    budget on an incomplete contrast. All eight assigned GPUs count during waits.
    """
    values = (used_gpu_hours, pair_gpu_hours, remaining_seconds, warmup_seconds, branch_seconds)
    if any(not math.isfinite(v) or v < 0 for v in values) or pair_gpu_hours == 0:
        raise ValueError("Admission requires finite measured costs and nonnegative remaining time")
    # Every later branch receives the slower first branch's runtime ceiling.
    # Reserving only the first pair's sum would undercount asymmetric methods.
    controller_seconds = 2 * (warmup_seconds * 1.25 + 60) + 4 * (branch_seconds * 1.25 + 60)
    reserve_gpu_hours = max(2 * pair_gpu_hours * 1.25, 8 * controller_seconds / 3600) + 32 * 300 / 3600
    reserve_seconds = 1.25 * (warmup_seconds + branch_seconds) + 300
    return {
        "admitted": used_gpu_hours + reserve_gpu_hours <= 32 and reserve_seconds <= remaining_seconds,
        "used_gpu_hours": used_gpu_hours,
        "reserved_gpu_hours": reserve_gpu_hours,
        "reserved_seconds": reserve_seconds,
        "remaining_seconds": remaining_seconds,
    }
