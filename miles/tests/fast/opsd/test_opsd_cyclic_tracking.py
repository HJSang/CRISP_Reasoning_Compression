"""Validation curves and cyclic display names preserve their scientific identity."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.utils.tracking_utils import opsd_wandb, wandb_utils


def test_validation_metrics_have_their_own_axis_and_exact_dataset():
    record = dict(completions=1024, responses_per_second=2, output_tokens_per_second=8000,
                  mean_response_tokens=4000,
                  metrics={"Math validation": dict.fromkeys(opsd_wandb.EVAL_FIELDS, 0.25)})
    payload = opsd_wandb.evaluation_metrics(record, completed_updates=7, suite="validation")
    assert payload["validation/step"] == 7
    assert payload["validation/Math_validation/avg_at_8"] == 0.25
    assert all(key.startswith("validation/") for key in payload)
    with pytest.raises(ValueError, match="Unexpected dataset"):
        opsd_wandb.evaluation_metrics(record, completed_updates=7)
    with pytest.raises(ValueError, match="Unexpected OPSD evaluation suite"):
        opsd_wandb.evaluation_metrics(record, completed_updates=7, suite="other")


@pytest.mark.parametrize("policy,group,label", [
    ("fixed_original", "fixed", "fixed original teacher"),
    ("cycle_refresh", "refresh", "cycle-refreshed teacher"),
])
def test_cyclic_name_uses_validated_phase_counts_and_teacher(policy, group, label):
    recipe = dict(target="frozen", context="worked", seed=137, task="math",
                  group=f"mathcycles-qwen3-4b-{group}-s137-r1", cyclic_teacher_policy=policy,
                  cyclic_pi_updates=7, cyclic_opd_updates=4, planned_updates=88)
    name = opsd_wandb.run_name(**recipe)
    assert f"Qwen3-4B | math | {label}" in name and "PI7 → OPD4 × 8" in name
    assert "fresh Adam per phase | constant LR | seed 137" in name
    for change in ({"seed": 138}, {"context": "none"}, {"ema_decay": 0.9},
                   {"cyclic_teacher_policy": "unexpected"}, {"task": "code"},
                   {"planned_updates": 87}, {"cyclic_pi_updates": 0},
                   {"group": "mathcycles-private-unrecognized"}):
        with pytest.raises(ValueError, match="Cyclic"):
            opsd_wandb.run_name(**(recipe | change))


def test_primary_initializer_forwards_cyclic_recipe_and_defines_validation_axis(monkeypatch):
    initialized, defined = [], []
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kw: initialized.append(kw))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *a, **kw: defined.append((a, kw)))
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="cycle-test"))
    args = Namespace(
        use_wandb=True, wandb_opsd_profile=True, wandb_mode="online", wandb_key=None,
        wandb_random_suffix=False, wandb_group="mathcycles-qwen3-8b-refresh-s137-r1",
        wandb_team="team", wandb_project="opsd", wandb_run_id="cycle-test", wandb_dir=None,
        seed=137, save=None, save_hf=None, num_rollout=88, opsd_target="frozen", opsd_context="worked",
        opsd_cyclic_teacher_policy="cycle_refresh", opsd_cyclic_pi_updates=7, opsd_cyclic_opd_updates=4,
    )
    wandb_utils.init_wandb_primary(args)
    assert "Qwen3-8B | math | cycle-refreshed teacher" in initialized[0]["name"]
    assert initialized[0]["config"]["opsd_cyclic_pi_updates"] == 7
    assert (("validation/*",), {"step_metric": "validation/step"}) in defined


@pytest.mark.parametrize("policy,label", [("reset_each_phase", "fresh Adam per phase"), ("carry", "carried Adam")])
@pytest.mark.parametrize("schedule,label_lr", [("constant", "constant LR"), ("global_linear", "global linear LR")])
def test_ablation_names_and_restricted_fields_describe_both_factors(policy, label, schedule, label_lr):
    optimizer_id = "reset" if policy == "reset_each_phase" else "carry"
    schedule_id = "constant" if schedule == "constant" else "linear"
    recipe = dict(target="frozen", context="worked", seed=137, cyclic_teacher_policy="fixed_original",
                  cyclic_pi_updates=7, cyclic_opd_updates=4, planned_updates=88,
                  cyclic_optimizer_policy=policy, cyclic_lr_schedule=schedule,
                  group=f"mathcycles-qwen3-4b-fixed-{optimizer_id}-{schedule_id}-s137-r1")
    name = opsd_wandb.run_name(**recipe)
    assert label in name and label_lr in name
    with pytest.raises(ValueError, match="recipe"):
        opsd_wandb.run_name(**(recipe | {"cyclic_optimizer_policy": "carry" if policy == "reset_each_phase" else "reset_each_phase"}))
    ready = recipe | {"group": recipe["group"].replace("-s137", "-ready-s137")}
    assert opsd_wandb.run_name(**ready).endswith(" | readiness")
    args = Namespace(opsd_cyclic_optimizer_policy=policy, opsd_cyclic_lr_schedule=schedule,
                     min_lr=5e-7, lr_decay_iters=87, raw_state="private")
    assert opsd_wandb.config(args) == {
        "opsd_cyclic_optimizer_policy": policy, "opsd_cyclic_lr_schedule": schedule,
        "min_lr": 5e-7, "lr_decay_iters": 87,
    }
    values = {"train/cyclic_applied_lr": 5e-6, "train/cyclic_next_lr": 4e-6,
              "train/cyclic_optimizer_updates": 8}
    assert opsd_wandb.metrics(values | {"optimizer_state": "private"}) == values
