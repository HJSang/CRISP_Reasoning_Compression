"""Pure, finite phase-length study recipes; no process, file, or admission operations."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from miles.utils.opsd_cyclic import cyclic_evaluation_suites

VERSION = "math-phases-v1"
MODELS = ("Qwen3-8B", "Qwen3-4B")
SPLITS = ((3, 8), (5, 6), (7, 4), (9, 2))
WORKERS = ("worker-a", "worker-b", "worker-c", "worker-d")
WAVES = (((3, 8), (7, 4)), ((5, 6), (9, 2)))
DATA_KEYS = frozenset({
    "train_manifest", "train_stream", "validation_panel", "validation_prompts",
    "validation_labels", "benchmark_prompts", "benchmark_labels", "plans_manifest",
})


@dataclass(frozen=True)
class PhaseRecipe:
    model_name: str
    pi_updates: int
    opd_updates: int
    updates: int = 88
    seed: int = 137
    teacher_policy: str = "cycle_refresh"
    optimizer_policy: str = "reset_each_phase"
    lr_schedule: str = "constant"
    learning_rate: float = 5e-6

    def __post_init__(self):
        counts = (self.pi_updates, self.opd_updates, self.updates, self.seed)
        if any(type(value) is not int for value in counts):
            raise ValueError("Phase lengths, horizon and seed must be integers, never booleans")
        if min(self.pi_updates, self.opd_updates, self.updates) < 1 or self.updates % (self.pi_updates + self.opd_updates):
            raise ValueError("Phase study requires positive lengths and complete cycles")
        if (self.model_name not in MODELS or (self.pi_updates, self.opd_updates) not in SPLITS
                or self.updates != 88 or self.seed != 137):
            raise ValueError("Recipe is outside the finite eight-arm phase study")
        if (self.teacher_policy != "cycle_refresh" or self.optimizer_policy != "reset_each_phase"
                or self.lr_schedule != "constant" or type(self.learning_rate) is not float or self.learning_rate != 5e-6):
            raise ValueError("Phase comparison fixes teacher refresh, reset Adam and constant learning rate")

    def run_id(self) -> str:
        size = {"Qwen3-4B": "4b", "Qwen3-8B": "8b"}[self.model_name]
        return (f"mathphases-v1-qwen3-{size}-refresh-pi{self.pi_updates}-opd{self.opd_updates}"
                "-u88-reset-constant-s137-r1")


def parse_run_id(run_id: str) -> PhaseRecipe:
    if not isinstance(run_id, str):
        raise ValueError("Phase-study identity must be a string")
    match = re.fullmatch(
        r"mathphases-v1-qwen3-(4b|8b)-refresh-pi([0-9]+)-opd([0-9]+)-u88-reset-constant-s137-r1",
        run_id,
    )
    if match is None:
        raise ValueError("Unknown phase-study identity")
    size, pi, opd = match.groups()
    recipe = PhaseRecipe(f"Qwen3-{size.upper()}", int(pi), int(opd))
    if recipe.run_id() != run_id:
        raise ValueError("Phase-study identity must use its canonical spelling")
    return recipe


def evaluation_schedule(recipe: PhaseRecipe) -> dict[int, tuple[str, ...]]:
    return {
        step: suites for step in range(recipe.updates + 1)
        if (suites := cyclic_evaluation_suites(
            step, planned_updates=recipe.updates, pi_updates=recipe.pi_updates, opd_updates=recipe.opd_updates
        ))
    }


def recipe_counts(recipe: PhaseRecipe) -> dict[str, int]:
    """Count trained evaluations only; shared originals are added once at study scope."""
    schedule = evaluation_schedule(recipe)
    validation = sum("validation" in suites for step, suites in schedule.items() if step)
    benchmark = sum("benchmark" in suites for step, suites in schedule.items() if step)
    return dict(updates=recipe.updates, cycles=recipe.updates // (recipe.pi_updates + recipe.opd_updates),
                training_questions=recipe.updates * 4, snapshot_jobs=len(schedule) - 1,
                validation_suites=validation, benchmark_suites=benchmark,
                suite_evaluations=validation + benchmark,
                generated_responses=validation * 128 * 8 + benchmark * 100 * 8)


def phase_cells(*, wave: int | None = None) -> list[dict]:
    if wave is not None and (type(wave) is not int or wave not in (1, 2)):
        raise ValueError("Only phase waves one and two exist")
    cells = []
    for number, splits in enumerate(WAVES, start=1):
        if wave is not None and wave != number:
            continue
        for index, worker in enumerate(WORKERS):
            recipe = PhaseRecipe(MODELS[index % 2], *splits[index // 2])
            baseline = PhaseRecipe(recipe.model_name, 3, 8).run_id()
            cells.append(dict(wave=number, worker=worker, run_id=recipe.run_id(), recipe=asdict(recipe),
                              generate_baseline=number == 1 and index < 2, baseline_source_run_id=baseline,
                              baseline_import_namespace=f"{VERSION}-wave-{number}-{worker}-baseline",
                              schedule=[dict(completed_updates=step, suites=list(suites))
                                        for step, suites in evaluation_schedule(recipe).items()],
                              counts=recipe_counts(recipe)))
    return cells


def study_counts() -> dict[str, int]:
    cells = phase_cells()
    producers = sum(cell["generate_baseline"] for cell in cells)
    validation = sum(cell["counts"]["validation_suites"] for cell in cells) + producers
    benchmark = sum(cell["counts"]["benchmark_suites"] for cell in cells) + producers
    return dict(trajectories=len(cells), fresh_baseline_models=producers,
                optimizer_updates=sum(cell["counts"]["updates"] for cell in cells),
                shared_training_questions=88 * 4, total_training_question_exposures=88 * 4 * len(cells),
                temporary_snapshots=sum(cell["counts"]["snapshot_jobs"] for cell in cells),
                generated_validation_suites=validation, generated_benchmark_suites=benchmark,
                generated_suites=validation + benchmark,
                generated_responses=validation * 128 * 8 + benchmark * 100 * 8,
                hosted_suite_writes=sum(cell["counts"]["suite_evaluations"] + 2 for cell in cells))


def stable_digest(value: dict) -> str:
    """Hash canonical JSON; NaN and infinity are not valid protocol evidence."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_hash(value: str):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Source identities require supplied lowercase SHA-256 digests")


def _validate_hashes(values: dict[str, str], keys: set[str] | frozenset[str]):
    if not isinstance(values, dict) or set(values) != keys:
        raise ValueError("Source hash mapping has missing or unexpected identities")
    for value in values.values():
        _validate_hash(value)


def build_manifest(*, model_sha256: dict[str, str], runtime_sha256: str, data_sha256: dict[str, str]) -> dict:
    """Bind supplied hashes without claiming to inspect their files or authorize execution."""
    _validate_hashes(model_sha256, set(MODELS))
    _validate_hash(runtime_sha256)
    _validate_hashes(data_sha256, DATA_KEYS)
    manifest = dict(version=VERSION, admitted=False, dispatch=False, historical_run_reuse=False,
                    sources=dict(model_sha256=dict(model_sha256), runtime_sha256=runtime_sha256,
                                 data_sha256=dict(data_sha256)),
                    cells=phase_cells(), counts=study_counts())
    return manifest | dict(manifest_sha256=stable_digest(manifest))


def validate_manifest(manifest: dict, *, expected_sha256: str):
    """Reject recipe/count/source tampering against an independently retained digest."""
    _validate_hash(expected_sha256)
    if not isinstance(manifest, dict) or set(manifest) != {
        "version", "admitted", "dispatch", "historical_run_reuse", "sources", "cells", "counts", "manifest_sha256"
    }:
        raise ValueError("Phase manifest has missing or unexpected fields")
    sources = manifest["sources"]
    if not isinstance(sources, dict) or set(sources) != {"model_sha256", "runtime_sha256", "data_sha256"}:
        raise ValueError("Phase manifest requires every explicit source binding")
    expected = build_manifest(model_sha256=sources["model_sha256"], runtime_sha256=sources["runtime_sha256"],
                              data_sha256=sources["data_sha256"])
    if stable_digest(manifest) != stable_digest(expected) or manifest["manifest_sha256"] != expected_sha256:
        raise ValueError("Phase manifest differs from the finite recipe or pinned source identity")
