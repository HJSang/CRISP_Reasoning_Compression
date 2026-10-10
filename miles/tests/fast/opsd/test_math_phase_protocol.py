"""Finite phase-study identities, accounting and scheduler boundaries on CPU."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from miles.utils.opsd_cyclic import cyclic_learning_rate, cyclic_optimizer_updates, cyclic_step
from tools.opsd.math_phase_protocol import (
    DATA_KEYS, MODELS, SPLITS, PhaseRecipe, build_manifest, evaluation_schedule,
    parse_run_id, phase_cells, recipe_counts, stable_digest, study_counts, validate_manifest,
)


def _manifest():
    digest = lambda name: hashlib.sha256(("synthetic:" + name).encode()).hexdigest()
    return build_manifest(model_sha256={model: digest(model) for model in MODELS},
                          runtime_sha256=digest("runtime"), data_sha256={name: digest(name) for name in DATA_KEYS})


@pytest.mark.parametrize("pi,opd", SPLITS)
def test_phase_boundaries_reset_counts_and_teacher_sources_follow_the_existing_core(pi, opd):
    recipe = PhaseRecipe("Qwen3-4B", pi, opd)
    args = SimpleNamespace(num_rollout=88, opsd_cyclic_pi_updates=pi, opsd_cyclic_opd_updates=opd,
                           opsd_cyclic_teacher_policy=recipe.teacher_policy,
                           opsd_cyclic_optimizer_policy=recipe.optimizer_policy,
                           opsd_cyclic_lr_schedule=recipe.lr_schedule, lr=recipe.learning_rate)
    resets, ends = [], []
    for update in range(88):
        cycle, offset = divmod(update, 11)
        step = cyclic_step(args, update)
        assert step.context == ("worked" if offset < pi else "none")
        assert step.cycle == cycle + 1 and step.teacher_source_update == cycle * 11
        assert step.phase_start == step.optimizer_reset == (offset in (0, pi))
        if step.optimizer_reset:
            resets.append(update)
        if step.cycle_end:
            ends.append(update + 1)
        assert cyclic_optimizer_updates(args, update + 1) == (offset + 1 if offset < pi else offset - pi + 1)
        assert cyclic_learning_rate(args, update) == 5e-6
    assert resets == [cycle * 11 + offset for cycle in range(8) for offset in (0, pi)]
    assert ends == list(range(11, 89, 11))


@pytest.mark.parametrize("pi,opd", SPLITS)
def test_schedule_counts_collisions_and_common_cycle_endpoints(pi, opd):
    recipe = PhaseRecipe("Qwen3-8B", pi, opd)
    schedule = evaluation_schedule(recipe)
    assert [step for step, suites in schedule.items() if "validation" in suites] == [
        0, *[cycle * 11 + offset for cycle in range(8) for offset in (pi, 11)],
    ]
    assert [step for step, suites in schedule.items() if "benchmark" in suites] == [
        0, 9, 18, 27, 36, 44, 53, 62, 71, 80, 88,
    ]
    assert schedule[0] == schedule[88] == ("validation", "benchmark")
    assert sum(len(suites) == 2 for step, suites in schedule.items() if step) == 4
    assert recipe_counts(recipe) == dict(updates=88, cycles=8, training_questions=352,
        snapshot_jobs=22, validation_suites=16, benchmark_suites=10, suite_evaluations=26, generated_responses=24384)


def test_two_waves_have_fresh_controls_and_only_two_original_baseline_generations():
    cells = phase_cells()
    assert len(cells) == len({cell["run_id"] for cell in cells}) == 8
    assert [(cell["worker"], cell["recipe"]["model_name"], cell["recipe"]["pi_updates"]) for cell in cells] == [
        ("worker-a", "Qwen3-8B", 3), ("worker-b", "Qwen3-4B", 3),
        ("worker-c", "Qwen3-8B", 7), ("worker-d", "Qwen3-4B", 7),
        ("worker-a", "Qwen3-8B", 5), ("worker-b", "Qwen3-4B", 5),
        ("worker-c", "Qwen3-8B", 9), ("worker-d", "Qwen3-4B", 9),
    ]
    producers = {cell["run_id"]: cell for cell in cells if cell["generate_baseline"]}
    assert len(producers) == 2 and len({cell["baseline_import_namespace"] for cell in cells}) == 8
    for cell in cells:
        source = producers[cell["baseline_source_run_id"]]
        assert source["recipe"]["model_name"] == cell["recipe"]["model_name"]
        assert parse_run_id(cell["run_id"]) == PhaseRecipe(**cell["recipe"])
        assert cell["run_id"].startswith("mathphases-v1-") and cell["run_id"].endswith("-r1")
    assert phase_cells(wave=1) + phase_cells(wave=2) == cells
    assert study_counts() == dict(trajectories=8, fresh_baseline_models=2, optimizer_updates=704,
        shared_training_questions=352, total_training_question_exposures=2816, temporary_snapshots=176,
        generated_validation_suites=130, generated_benchmark_suites=82, generated_suites=212,
        generated_responses=198720, hosted_suite_writes=224)


@pytest.mark.parametrize("changes", [
    {"pi_updates": True}, {"opd_updates": 8.0}, {"updates": 88.0}, {"seed": True},
    {"pi_updates": 0}, {"pi_updates": -1}, {"pi_updates": 2, "opd_updates": 9},
    {"updates": 87}, {"updates": 176}, {"seed": 138}, {"model_name": "Qwen3-1.7B"},
    {"teacher_policy": "fixed_original"}, {"optimizer_policy": "carry"},
    {"lr_schedule": "global_linear"}, {"learning_rate": float("nan")}, {"learning_rate": True},
])
def test_recipe_rejects_invalid_types_partial_cycles_and_out_of_scope_changes(changes):
    with pytest.raises(ValueError):
        PhaseRecipe(**(dict(model_name="Qwen3-4B", pi_updates=3, opd_updates=8) | changes))


@pytest.mark.parametrize("token,replacement", [
    ("pi3", "pi03"), ("pi3", "pi4"), ("opd8", "opd7"), ("u88", "u176"),
    ("reset", "carry"), ("constant", "linear"), ("s137", "s138"), ("r1", "r2"),
])
def test_identity_cannot_disagree_with_its_finite_recipe(token, replacement):
    identity = PhaseRecipe("Qwen3-4B", 3, 8).run_id()
    with pytest.raises(ValueError):
        parse_run_id(identity.replace(token, replacement))


@pytest.mark.parametrize("wave", [True, 1.0, 0, 3])
def test_invalid_wave_cannot_expand_the_matrix(wave):
    with pytest.raises(ValueError):
        phase_cells(wave=wave)


def test_manifest_roundtrip_pins_supplied_sources_and_does_not_admit_or_dispatch():
    manifest = _manifest()
    expected = manifest["manifest_sha256"]
    validate_manifest(json.loads(json.dumps(manifest)), expected_sha256=expected)
    assert manifest["admitted"] is False and manifest["dispatch"] is False
    assert stable_digest(dict(reversed(list(manifest.items())))) == stable_digest(manifest)
    assert manifest["historical_run_reuse"] is False


@pytest.mark.parametrize("change", ["recipe", "identity", "baseline", "count", "boolean", "source", "model_source", "admission"])
def test_manifest_rejects_tampering_even_with_a_recomputed_internal_digest(change):
    manifest = _manifest()
    expected = manifest["manifest_sha256"]
    if change == "recipe":
        manifest["cells"][0]["recipe"]["pi_updates"] = 5
    elif change == "identity":
        manifest["cells"][0]["run_id"] = "mathcycles-qwen3-8b-refresh-s137-r2"
    elif change == "baseline":
        manifest["cells"][4]["baseline_source_run_id"] = manifest["cells"][1]["run_id"]
    elif change == "count":
        manifest["counts"]["generated_responses"] -= 1824
    elif change == "boolean":
        manifest["cells"][0]["wave"] = True
    elif change == "source":
        manifest["sources"]["runtime_sha256"] = "a" * 64
    elif change == "model_source":
        models = manifest["sources"]["model_sha256"]
        models["Qwen3-4B"], models["Qwen3-8B"] = models["Qwen3-8B"], models["Qwen3-4B"]
    else:
        manifest["admitted"] = True
    manifest["manifest_sha256"] = stable_digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    with pytest.raises(ValueError):
        validate_manifest(manifest, expected_sha256=expected)
    if change not in {"source", "model_source"}:
        with pytest.raises(ValueError):
            validate_manifest(manifest, expected_sha256=manifest["manifest_sha256"])


@pytest.mark.parametrize("bad", [None, True, "", "a" * 63, "A" * 64, "g" * 64])
def test_source_hashes_must_be_explicit_valid_sha256(bad):
    sources = copy.deepcopy(_manifest()["sources"])
    sources["runtime_sha256"] = bad
    with pytest.raises(ValueError):
        build_manifest(**sources)


def test_source_maps_require_all_models_and_data_roles_and_are_copied():
    sources = _manifest()["sources"]
    manifest = build_manifest(**sources)
    sources["data_sha256"]["train_stream"] = "a" * 64
    assert manifest["sources"]["data_sha256"]["train_stream"] != "a" * 64
    del sources["data_sha256"]["validation_labels"]
    with pytest.raises(ValueError):
        build_manifest(**sources)
    sources = _manifest()["sources"]
    sources["model_sha256"]["unknown"] = "a" * 64
    with pytest.raises(ValueError):
        build_manifest(**sources)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_cannot_be_canonical_evidence(value):
    with pytest.raises(ValueError):
        stable_digest({"value": value})
