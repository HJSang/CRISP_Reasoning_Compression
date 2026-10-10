"""Cyclic scheduling, source promotion and fresh-Adam boundaries on CPU."""

import sys
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.opsd_cyclic import begin_cyclic_step, complete_cyclic_step, initialize_cyclic
from miles.utils.opsd_cyclic import cyclic_evaluation_suites, cyclic_learning_rate, cyclic_optimizer_updates, cyclic_step


def _args(policy="cycle_refresh"):
    return SimpleNamespace(
        opsd_cyclic_teacher_policy=policy, opsd_cyclic_pi_updates=7, opsd_cyclic_opd_updates=4,
        num_rollout=88, stream_optimizer_state_to_disk=False, chunked_optimizer_state_offload=False,
        opsd_cyclic_optimizer_policy="reset_each_phase", opsd_cyclic_lr_schedule="constant",
        lr=5e-6, min_lr=5e-7, global_batch_size=4,
    )


class _Backuper:
    def __init__(self):
        self.weights = {"actor": torch.tensor([3.0]), "teacher": torch.tensor([1.0])}
        self.copies = []

    def copy(self, *, src_tag, dst_tag):
        self.weights[dst_tag].copy_(self.weights[src_tag].detach())
        self.copies.append((src_tag, dst_tag))


def _actor(policy="cycle_refresh"):
    actor = SimpleNamespace(args=_args(policy), weights_backuper=_Backuper(),
                            optimizer=SimpleNamespace(param_groups=[{"lr": 5e-6}]),
                            opt_param_scheduler=SimpleNamespace(num_steps=0))
    initialize_cyclic(actor)
    return actor


def _rollout(context):
    return {"tokens": [torch.tensor([1, 2, 3])], "response_lengths": [1],
            "teacher_prompt_ids": [[1, 2] if context == "none" else [5, 6]]}


def _mock_reset(monkeypatch):
    resets = []
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.optimizer_utils", SimpleNamespace(
        reset_optimizer_state=lambda optimizer, **kwargs: resets.append((optimizer, kwargs))
    ))
    return resets


@pytest.mark.parametrize("policy", ["fixed_original", "cycle_refresh"])
def test_all_eight_cycles_keep_teacher_through_pi_and_opd_and_reset_only_at_boundaries(monkeypatch, policy):
    resets = _mock_reset(monkeypatch)
    actor = _actor(policy)
    initial_teacher = 1 if policy == "fixed_original" else 3
    reset_steps = []
    for rollout_id in range(88):
        step = cyclic_step(actor.args, rollout_id)
        expected_teacher = initial_teacher if policy == "fixed_original" else 3 + (rollout_id // 11) * 11
        assert actor.weights_backuper.weights["teacher"].item() == expected_teacher
        before_resets = len(resets)
        begin_cyclic_step(actor, _rollout(step.context), rollout_id=rollout_id, num_optimizer_steps=1)
        if len(resets) != before_resets:
            reset_steps.append(rollout_id)
        actor.weights_backuper.weights["actor"].add_(1)
        actor.opt_param_scheduler.num_steps += 4
        assert actor.weights_backuper.weights["teacher"].item() == expected_teacher
        metrics = complete_cyclic_step(actor, rollout_id=rollout_id)
        assert metrics["train/cyclic_completed_updates"] == rollout_id + 1
        expected_source = (rollout_id // 11) * 11 if policy == "cycle_refresh" else 0
        assert metrics["train/cyclic_teacher_source_update"] == expected_source
        assert metrics["train/cyclic_teacher_refreshed"] == int(policy == "cycle_refresh" and rollout_id % 11 == 10)
    assert reset_steps == [cycle * 11 + offset for cycle in range(8) for offset in (0, 7)]
    assert actor.weights_backuper.weights["actor"].item() == 91
    assert actor.weights_backuper.weights["teacher"].item() == (91 if policy == "cycle_refresh" else 1)
    assert len(actor.weights_backuper.copies) == (9 if policy == "cycle_refresh" else 0)


def test_rejected_or_missing_update_does_not_promote_teacher(monkeypatch):
    _mock_reset(monkeypatch)
    actor = _actor()
    begin_cyclic_step(actor, _rollout("worked"), rollout_id=0, num_optimizer_steps=1)
    # A failed train never calls completion. Advancing its rollout ID must fail.
    with pytest.raises(ValueError, match="sequential"):
        begin_cyclic_step(actor, _rollout("worked"), rollout_id=1, num_optimizer_steps=1)
    assert actor.opsd_cyclic_completed_updates == 0
    assert len(actor.weights_backuper.copies) == 1
    with pytest.raises(ValueError, match="out-of-order"):
        complete_cyclic_step(actor, rollout_id=1)


@pytest.mark.parametrize("rollout_id,context", [(0, "none"), (7, "worked")])
def test_teacher_prefix_must_match_phase_before_optimizer_reset(monkeypatch, rollout_id, context):
    resets = _mock_reset(monkeypatch)
    actor = _actor()
    actor.opsd_cyclic_completed_updates = rollout_id
    with pytest.raises(ValueError, match="prefix"):
        begin_cyclic_step(actor, _rollout(context), rollout_id=rollout_id, num_optimizer_steps=1)
    assert not resets


def test_stale_cycle_teacher_and_multiple_updates_are_rejected(monkeypatch):
    _mock_reset(monkeypatch)
    actor = _actor()
    actor.opsd_cyclic_completed_updates = 11
    with pytest.raises(ValueError, match="cycle-entry"):
        begin_cyclic_step(actor, _rollout("worked"), rollout_id=11, num_optimizer_steps=1)
    with pytest.raises(ValueError, match="one sequential"):
        begin_cyclic_step(actor, _rollout("worked"), rollout_id=11, num_optimizer_steps=2)


def test_evaluation_endpoints_are_independent_of_cycle_teacher_policy():
    schedules = {step: cyclic_evaluation_suites(step, planned_updates=88) for step in range(89)}
    assert [step for step, suites in schedules.items() if "benchmark" in suites] == [0, 9, 18, 27, 36, 44, 53, 62, 71, 80, 88]
    assert [step for step, suites in schedules.items() if "validation" in suites] == [0, *[cycle * 11 + offset for cycle in range(8) for offset in (7, 11)]]


@pytest.mark.parametrize("rollout_id", [None, -1, 88])
def test_missing_or_out_of_range_rollout_identity_fails(rollout_id):
    with pytest.raises(ValueError, match="in-range rollout ID"):
        cyclic_step(_args(), rollout_id)


@pytest.mark.parametrize("optimizer_policy", ["reset_each_phase", "carry"])
@pytest.mark.parametrize("schedule", ["constant", "global_linear"])
@pytest.mark.parametrize("updates,pi,opd", [(88, 7, 4), (4, 1, 1)])
def test_real_adam_and_megatron_scheduler_preserve_independent_clocks(monkeypatch, optimizer_policy, schedule, updates, pi, opd):
    optimizer_module = pytest.importorskip("megatron.core.optimizer.optimizer")
    scheduler_module = pytest.importorskip("megatron.core.optimizer_param_scheduler")
    reset_module = pytest.importorskip("miles.backends.megatron_utils.optimizer_utils")
    monkeypatch.setattr(reset_module, "USING_PYTORCH_OPTIMIZER", True)
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    inner = torch.optim.Adam([param], lr=5e-6, betas=(0.9, 0.999))
    chained = optimizer_module.ChainedOptimizer.__new__(optimizer_module.ChainedOptimizer)
    chained.config = SimpleNamespace(offload_optimizer_states=False, fp16=False)
    chained.chained_optimizers = [SimpleNamespace(
        optimizer=inner, param_groups=inner.param_groups, is_stub_optimizer=False,
        init_state_fn=None, zero_grad=inner.zero_grad,
    )]
    scheduler = scheduler_module.OptimizerParamScheduler(
        chained, init_lr=0, max_lr=5e-6, min_lr=5e-7, lr_warmup_steps=0,
        lr_decay_steps=(updates - 1) * 4, lr_decay_style="linear" if schedule == "global_linear" else "constant",
        start_wd=0, end_wd=0, wd_incr_steps=updates * 4, wd_incr_style="constant",
    )
    actor = _actor("fixed_original")
    actor.optimizer, actor.opt_param_scheduler = chained, scheduler
    actor.args.__dict__.update(opsd_cyclic_optimizer_policy=optimizer_policy, opsd_cyclic_lr_schedule=schedule,
                               num_rollout=updates, opsd_cyclic_pi_updates=pi, opsd_cyclic_opd_updates=opd)
    applied = []
    for k in range(updates):
        step = cyclic_step(actor.args, k)
        previous_state = {name: value.clone() for name, value in inner.state.get(param, {}).items()}
        begin_cyclic_step(actor, _rollout(step.context), rollout_id=k, num_optimizer_steps=1)
        if step.optimizer_reset:
            assert not inner.state and param.grad is None
        else:
            for name, value in previous_state.items():
                torch.testing.assert_close(inner.state[param][name], value, rtol=0, atol=0)
        expected_lr = 5e-6 if schedule == "constant" else 5e-6 * (1 - 0.9 * k / (updates - 1))
        assert inner.param_groups[0]["lr"] == pytest.approx(expected_lr)
        param.grad = torch.tensor([0.5, -0.25]) if step.context == "worked" else torch.tensor([-0.25, 0.5])
        inner.step()
        scheduler.step(increment=4)
        metrics = complete_cyclic_step(actor, rollout_id=k)
        applied.append(metrics["train/cyclic_applied_lr"])
        local_count = k + 1
        if optimizer_policy == "reset_each_phase":
            offset = k % (pi + opd)
            local_count = offset + 1 if offset < pi else offset - pi + 1
        assert inner.state[param]["step"].item() == metrics["train/cyclic_optimizer_updates"] == local_count
        assert cyclic_optimizer_updates(actor.args, k + 1) == local_count
        assert metrics["train/cyclic_optimizer_reset"] == int(step.optimizer_reset)
        assert metrics["train/cyclic_next_lr"] == inner.param_groups[0]["lr"]
        assert scheduler.num_steps == (k + 1) * 4
    assert applied[0] == pytest.approx(5e-6)
    assert applied[-1] == pytest.approx(5e-7 if schedule == "global_linear" else 5e-6)


def test_failed_update_cannot_advance_completion_or_hide_a_restarted_scheduler(monkeypatch):
    _mock_reset(monkeypatch)
    actor = _actor()
    begin_cyclic_step(actor, _rollout("worked"), rollout_id=0, num_optimizer_steps=1)
    with pytest.raises(ValueError, match="clock"):
        complete_cyclic_step(actor, rollout_id=0)
    assert actor.opsd_cyclic_completed_updates == actor.opsd_cyclic_optimizer_updates == 0
    actor.opsd_cyclic_completed_updates = 7
    with pytest.raises(ValueError, match="clock"):
        begin_cyclic_step(actor, _rollout("none"), rollout_id=7, num_optimizer_steps=1)
    actor.opt_param_scheduler.num_steps = 28
    actor.optimizer.param_groups[0]["lr"] = 1e-6
    with pytest.raises(ValueError, match="parameter-group LR"):
        begin_cyclic_step(actor, _rollout("none"), rollout_id=7, num_optimizer_steps=1)
