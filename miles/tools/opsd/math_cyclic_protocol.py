"""Pure math study cells and evaluation cadence; process ownership stays in the coordinator."""

from dataclasses import dataclass

from miles.utils.opsd_cyclic import cyclic_evaluation_suites


@dataclass(frozen=True)
class MathCycleCell:
    worker: str
    model_name: str
    teacher_policy: str
    baseline_worker: str
    optimizer_policy: str = "reset_each_phase"
    lr_schedule: str = "constant"
    study: str = "teacher_policy"

    def __post_init__(self):
        if self.optimizer_policy not in {"reset_each_phase", "carry"} or self.lr_schedule not in {"constant", "global_linear"}:
            raise ValueError("Unknown cyclic optimizer/LR recipe")
        if self.study not in {"teacher_policy", "optimizer_lr", "readiness"}:
            raise ValueError("Unknown cyclic study")
        if self.study == "teacher_policy" and (self.optimizer_policy != "reset_each_phase" or self.lr_schedule != "constant"):
            raise ValueError("Teacher-policy screen preserves reset Adam and constant LR")

    def run_id(self, *, seed: int = 137, attempt: int = 1) -> str:
        if seed < 0 or attempt < 1:
            raise ValueError("Seed and attempt must identify a nonnegative seed and positive attempt")
        size = {"Qwen3-4B": "4b", "Qwen3-8B": "8b"}[self.model_name]
        policy = {"fixed_original": "fixed", "cycle_refresh": "refresh"}[self.teacher_policy]
        recipe = ""
        if self.study != "teacher_policy":
            optimizer = {"reset_each_phase": "reset", "carry": "carry"}[self.optimizer_policy]
            schedule = {"constant": "constant", "global_linear": "linear"}[self.lr_schedule]
            recipe = f"-{optimizer}-{schedule}" + ("-ready" if self.study == "readiness" else "")
        return f"mathcycles-qwen3-{size}-{policy}{recipe}-s{seed}-r{attempt}"


def screen_cells() -> tuple[MathCycleCell, ...]:
    return (
        MathCycleCell("worker-a", "Qwen3-8B", "fixed_original", "worker-a"),
        MathCycleCell("worker-b", "Qwen3-4B", "fixed_original", "worker-b"),
        MathCycleCell("worker-c", "Qwen3-8B", "cycle_refresh", "worker-a"),
        MathCycleCell("worker-d", "Qwen3-4B", "cycle_refresh", "worker-b"),
    )


def optimizer_ablation_cells(model_name: str) -> tuple[MathCycleCell, ...]:
    """One model per four-worker factorial wave, with the original teacher held fixed."""
    if model_name not in {"Qwen3-4B", "Qwen3-8B"}:
        raise ValueError("Optimizer ablation supports Qwen3-4B or Qwen3-8B")
    recipes = (
        ("reset_each_phase", "constant"), ("carry", "constant"),
        ("reset_each_phase", "global_linear"), ("carry", "global_linear"),
    )
    return tuple(
        MathCycleCell(worker, model_name, "fixed_original", "worker-a", optimizer, schedule, "optimizer_lr")
        for worker, (optimizer, schedule) in zip(("worker-a", "worker-b", "worker-c", "worker-d"), recipes, strict=True)
    )


def evaluation_schedule(*, updates: int = 88, pi_updates: int = 7, opd_updates: int = 4) -> dict[int, tuple[str, ...]]:
    if min(updates, pi_updates, opd_updates) <= 0 or updates % (pi_updates + opd_updates):
        raise ValueError("Schedule requires positive phase lengths and complete cycles")
    return {
        step: suites
        for step in range(updates + 1)
        if (suites := cyclic_evaluation_suites(
            step, planned_updates=updates, pi_updates=pi_updates, opd_updates=opd_updates
        ))
    }


def screen_counts(*, validation_questions: int = 128) -> dict[str, int]:
    if validation_questions <= 0:
        raise ValueError("Validation panel must contain questions")
    schedule = evaluation_schedule()
    phase_evals = sum("validation" in kinds for step, kinds in schedule.items() if step)
    benchmark_evals = sum("benchmark" in kinds for step, kinds in schedule.items() if step)
    validation_total, benchmark_total = 4 * phase_evals + 2, 4 * benchmark_evals + 2
    return {
        "optimizer_updates": 4 * 88,
        "shared_training_questions": 88 * 4,
        "temporary_snapshots": 4 * (len(schedule) - 1),
        "validation_evaluations": validation_total,
        "benchmark_evaluations": benchmark_total,
        "validation_responses": validation_total * validation_questions * 8,
        "benchmark_responses": benchmark_total * 100 * 8,
        "total_responses": validation_total * validation_questions * 8 + benchmark_total * 100 * 8,
    }
