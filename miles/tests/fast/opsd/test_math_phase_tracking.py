"""Keep the finite phase identities inside the existing private-data tracking boundary."""

import pytest

from miles.utils.tracking_utils.opsd_wandb import run_name
from tools.opsd.math_phase_protocol import phase_cells


def naming(cell):
    recipe = cell["recipe"]
    return dict(target="frozen", context="worked", task="math", seed=137, group=cell["run_id"],
                cyclic_teacher_policy=recipe["teacher_policy"], cyclic_pi_updates=recipe["pi_updates"],
                cyclic_opd_updates=recipe["opd_updates"], planned_updates=88,
                cyclic_optimizer_policy=recipe["optimizer_policy"], cyclic_lr_schedule=recipe["lr_schedule"])


@pytest.mark.parametrize("cell", phase_cells(), ids=lambda cell: cell["run_id"])
def test_all_eight_phase_names_match_the_frozen_protocol(cell):
    label = run_name(**naming(cell))
    recipe = cell["recipe"]
    assert label.startswith(recipe["model_name"] + " | math | cycle-refreshed teacher")
    assert f'PI{recipe["pi_updates"]} → OPD{recipe["opd_updates"]} × 8' in label
    assert label.endswith("seed 137 | non-thinking | attempt 1 | phase study")
    assert "fresh Adam per phase | constant LR" in label


@pytest.mark.parametrize("key,value", [
    ("seed", 138), ("seed", 137.0), ("target", "current"), ("context", "none"), ("task", "code"),
    ("ema_decay", 0.9), ("cyclic_teacher_policy", "fixed_original"), ("cyclic_pi_updates", 5),
    ("cyclic_opd_updates", 6), ("planned_updates", 22), ("planned_updates", True),
    ("cyclic_optimizer_policy", "carry"), ("cyclic_lr_schedule", "global_linear"),
])
def test_phase_name_rejects_actual_argument_mismatch(key, value):
    with pytest.raises(ValueError, match="recipe"):
        run_name(**(naming(phase_cells()[0]) | {key: value}))


@pytest.mark.parametrize("old,new", [("pi3", "pi03"), ("opd8", "opd6"), ("r1", "r2"),
                                      ("s137", "s138"), ("8b", "1p7b"), ("v1", "v2")])
def test_unknown_phase_identity_cannot_fall_back_to_an_unvalidated_generic_name(old, new):
    args = naming(phase_cells()[0])
    args["group"] = args["group"].replace(old, new)
    with pytest.raises(ValueError):
        run_name(**args)


@pytest.mark.parametrize("size,pi,opd", [("4b", 3, 8), ("8b", 9, 2)])
def test_readiness_label_cannot_be_confused_with_an_eight_cycle_scientific_run(size, pi, opd):
    args = naming(phase_cells()[0]) | dict(cyclic_pi_updates=pi, cyclic_opd_updates=opd, planned_updates=22,
        group=f"mathphases-ready-v1-qwen3-{size}-refresh-pi{pi}-opd{opd}-u22-reset-constant-s137-r1")
    label = run_name(**args)
    assert f"PI{pi} → OPD{opd} × 2" in label and label.endswith("phase readiness")
    with pytest.raises(ValueError):
        run_name(**(args | {"planned_updates": 88}))


@pytest.mark.parametrize("size,pi,opd", [("8b", 3, 8), ("4b", 9, 2), ("4b", 7, 4)])
def test_readiness_namespace_does_not_expand_to_unapproved_models_or_phase_splits(size, pi, opd):
    args = naming(phase_cells()[0]) | dict(cyclic_pi_updates=pi, cyclic_opd_updates=opd, planned_updates=22,
        group=f"mathphases-ready-v1-qwen3-{size}-refresh-pi{pi}-opd{opd}-u22-reset-constant-s137-r1")
    with pytest.raises(ValueError):
        run_name(**args)


def test_existing_cycle_and_arbitrary_private_groups_keep_their_previous_behavior():
    args = naming(phase_cells()[0]) | dict(group="mathcycles-qwen3-8b-refresh-s137-r2",
                                         cyclic_pi_updates=7, cyclic_opd_updates=4)
    label = run_name(**args)
    assert label.endswith("non-thinking | attempt 2") and "phase study" not in label
    generic = run_name(target="frozen", context="worked", seed=137, group="/private/unpublished")
    assert generic == "frozen teacher | worked-solution PI | direct target | seed 137"
