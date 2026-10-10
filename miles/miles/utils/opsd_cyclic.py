"""Update-indexed PI/OPD phases; model and optimizer state remain owned by the actor."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CyclicStep:
    cycle: int
    context: str
    phase_start: bool
    cycle_end: bool
    teacher_source_update: int

    def metrics(self) -> dict[str, int]:
        return {
            "train/cyclic_cycle": self.cycle,
            "train/cyclic_pi_phase": int(self.context == "worked"),
            "train/cyclic_optimizer_reset": int(self.phase_start),
            "train/cyclic_teacher_source_update": self.teacher_source_update,
        }


def cyclic_step(args, rollout_id: int) -> CyclicStep:
    """Zero-based rollout IDs identify the student before its next optimizer update."""
    if not isinstance(rollout_id, int) or not 0 <= rollout_id < args.num_rollout:
        raise ValueError("Cyclic OPSD requires an in-range rollout ID")
    pi, opd = args.opsd_cyclic_pi_updates, args.opsd_cyclic_opd_updates
    cycle, offset = divmod(rollout_id, pi + opd)
    return CyclicStep(
        cycle=cycle + 1,
        context="worked" if offset < pi else "none",
        phase_start=offset in (0, pi),
        cycle_end=offset == pi + opd - 1,
        teacher_source_update=cycle * (pi + opd) if args.opsd_cyclic_teacher_policy == "cycle_refresh" else 0,
    )


def cyclic_evaluation_suites(
    completed_updates: int, *, planned_updates: int, pi_updates: int = 7, opd_updates: int = 4
) -> tuple[str, ...]:
    """Validation follows phase endpoints; benchmark cadence uses the planned horizon."""
    if not 0 <= completed_updates <= planned_updates:
        raise ValueError("Cyclic evaluation update lies outside the planned run")
    suites = []
    if completed_updates % (pi_updates + opd_updates) in (0, pi_updates):
        suites.append("validation")
    benchmark_steps = {0, *((k * planned_updates + 9) // 10 for k in range(1, 11))}
    if completed_updates in benchmark_steps:
        suites.append("benchmark")
    return tuple(suites)
