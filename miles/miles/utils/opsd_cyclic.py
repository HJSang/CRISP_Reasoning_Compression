"""Update-indexed PI/OPD phases; model and optimizer state remain owned by the actor."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CyclicStep:
    cycle: int
    context: str
    phase_start: bool
    optimizer_reset: bool
    cycle_end: bool
    teacher_source_update: int

    def metrics(self) -> dict[str, int]:
        return {
            "train/cyclic_cycle": self.cycle,
            "train/cyclic_pi_phase": int(self.context == "worked"),
            "train/cyclic_optimizer_reset": int(self.optimizer_reset),
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
        optimizer_reset=offset in (0, pi) and args.opsd_cyclic_optimizer_policy == "reset_each_phase",
        cycle_end=offset == pi + opd - 1,
        teacher_source_update=cycle * (pi + opd) if args.opsd_cyclic_teacher_policy == "cycle_refresh" else 0,
    )


def cyclic_optimizer_updates(args, completed_updates: int) -> int:
    """Successful updates since the last reset; carry includes every completed phase."""
    if completed_updates == 0 or args.opsd_cyclic_optimizer_policy == "carry":
        return completed_updates
    offset = (completed_updates - 1) % (args.opsd_cyclic_pi_updates + args.opsd_cyclic_opd_updates)
    return offset + 1 if offset < args.opsd_cyclic_pi_updates else offset - args.opsd_cyclic_pi_updates + 1


def cyclic_learning_rate(args, completed_updates: int) -> float:
    """LR for the next update; the last planned update uses the linear decay floor."""
    if args.opsd_cyclic_lr_schedule == "constant":
        return args.lr
    fraction = min(completed_updates / (args.num_rollout - 1), 1.0)
    return args.min_lr + (args.lr - args.min_lr) * (1.0 - fraction)


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
