"""Matched frozen/EMA PI-to-OPD screens; pure descriptions, no job launching."""

import math
from pathlib import PurePosixPath


def cell_jobs(*, root: str, model_name: str, task: str, base_sha256: str, seed: int = 101) -> list[dict]:
    if model_name not in {"Qwen3-1.7B", "Qwen3-8B"} or task not in {"math", "code"} or seed != 101:
        raise ValueError("Use the approved model/task cells and screening seed")
    root = PurePosixPath(root)
    size = "1p7b" if model_name == "Qwen3-1.7B" else "8b"
    jobs = []
    for policy in ("frozen", "ema"):
        stem = f"generality-qwen3-{size}-{task}-{policy}"
        warmup_id = f"{stem}-warmup-s{seed}-r1"
        source = root / "runs" / warmup_id / "eval-snapshots" / "step_6"
        decay = 0.9 if policy == "ema" else None
        jobs.append(
            dict(
                run_id=warmup_id,
                context="worked",
                updates=7,
                dataset=f"{task}-warmup.jsonl",
                seed=seed,
                snapshot_interval=7,
                retain_final_snapshot=True,
                expected_evaluations=1,
                initial_checkpoint=str(root / "models" / model_name),
                initial_sha256=base_sha256,
                ema_decay=decay,
                expected_ema_updates=7 if decay is not None else None,
            )
        )
        for phase in ("pi", "opd"):
            continuing_ema = phase == "pi" and decay is not None
            jobs.append(
                dict(
                    run_id=f"{stem}-{phase}-s{seed}-r1",
                    context="worked" if phase == "pi" else "none",
                    updates=4,
                    dataset=f"{task}-adaptation.jsonl",
                    seed=seed,
                    snapshot_interval=1,
                    retain_final_snapshot=False,
                    expected_evaluations=4,
                    initial_checkpoint=str(source),
                    parent_run=warmup_id,
                    parent_updates=7,
                    ema_decay=decay if continuing_ema else None,
                    ema_source=str(source / "ema-teacher") if continuing_ema else None,
                    expected_ema_updates=11 if continuing_ema else None,
                )
            )
    return jobs


def admit_cells(reservations: list[float], *, remaining_gpu_hours: float, remaining_seconds: float) -> bool:
    """Admit all four complete cells together, including their evaluation/cleanup reserve."""
    values = [*reservations, remaining_gpu_hours, remaining_seconds]
    if len(reservations) != 4 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Admission needs four positive measured reservations and finite remaining limits")
    # Each independent worker holds all eight assigned GPUs for its full interval.
    return sum(reservations) <= remaining_gpu_hours and max(reservations) * 3600 / 8 <= remaining_seconds
