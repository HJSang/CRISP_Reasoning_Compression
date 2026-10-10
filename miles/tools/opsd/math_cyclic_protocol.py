"""Pure four-cell math screen definitions; process ownership stays in the coordinator."""

from dataclasses import dataclass

from miles.utils.opsd_cyclic import cyclic_evaluation_suites


@dataclass(frozen=True)
class MathCycleCell:
    worker: str
    model_name: str
    teacher_policy: str
    baseline_worker: str

    def run_id(self, *, seed: int = 137, attempt: int = 1) -> str:
        if seed < 0 or attempt < 1:
            raise ValueError("Seed and attempt must identify a nonnegative seed and positive attempt")
        size = {"Qwen3-4B": "4b", "Qwen3-8B": "8b"}[self.model_name]
        policy = {"fixed_original": "fixed", "cycle_refresh": "refresh"}[self.teacher_policy]
        return f"mathcycles-qwen3-{size}-{policy}-s{seed}-r{attempt}"


def screen_cells() -> tuple[MathCycleCell, ...]:
    return (
        MathCycleCell("worker-a", "Qwen3-8B", "fixed_original", "worker-a"),
        MathCycleCell("worker-b", "Qwen3-4B", "fixed_original", "worker-b"),
        MathCycleCell("worker-c", "Qwen3-8B", "cycle_refresh", "worker-a"),
        MathCycleCell("worker-d", "Qwen3-4B", "cycle_refresh", "worker-b"),
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
