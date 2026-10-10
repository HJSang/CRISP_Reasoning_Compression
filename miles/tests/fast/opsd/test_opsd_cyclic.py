"""Cyclic scheduling, source promotion and fresh-Adam boundaries on CPU."""

import sys
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.opsd_cyclic import begin_cyclic_step, complete_cyclic_step, initialize_cyclic
from miles.utils.opsd_cyclic import cyclic_evaluation_suites, cyclic_step


def _args(policy="cycle_refresh"):
    return SimpleNamespace(
        opsd_cyclic_teacher_policy=policy, opsd_cyclic_pi_updates=7, opsd_cyclic_opd_updates=4,
        num_rollout=88, stream_optimizer_state_to_disk=False, chunked_optimizer_state_offload=False,
    )


class _Backuper:
    def __init__(self):
        self.weights = {"actor": torch.tensor([3.0]), "teacher": torch.tensor([1.0])}
        self.copies = []

    def copy(self, *, src_tag, dst_tag):
        self.weights[dst_tag].copy_(self.weights[src_tag].detach())
        self.copies.append((src_tag, dst_tag))


def _actor(policy="cycle_refresh"):
    actor = SimpleNamespace(args=_args(policy), weights_backuper=_Backuper(), optimizer=object())
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
