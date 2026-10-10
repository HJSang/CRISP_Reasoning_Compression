"""Pairing and evaluation coverage for the fixed versus refreshed teacher screen."""

import json
import shlex
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.utils.opsd_study import enqueue_evaluation
from tools.opsd.math_cyclic_protocol import evaluation_schedule, screen_cells, screen_counts


def test_balanced_cells_share_only_an_identical_model_baseline():
    cells = screen_cells()
    assert len({cell.run_id() for cell in cells}) == 4
    for model in {cell.model_name for cell in cells}:
        pair = [cell for cell in cells if cell.model_name == model]
        assert {cell.teacher_policy for cell in pair} == {"fixed_original", "cycle_refresh"}
        assert len({cell.baseline_worker for cell in pair}) == 1
        assert pair[0].baseline_worker in {cell.worker for cell in pair}


def test_every_phase_endpoint_and_percentage_milestone_is_reserved():
    schedule = evaluation_schedule()
    assert [s for s, kinds in schedule.items() if "benchmark" in kinds] == [
        0, 9, 18, 27, 36, 44, 53, 62, 71, 80, 88,
    ]
    assert [s for s, kinds in schedule.items() if "validation" in kinds] == [
        0, 7, 11, 18, 22, 29, 33, 40, 44, 51, 55, 62, 66, 73, 77, 84, 88,
    ]
    assert schedule[18] == ("validation", "benchmark")
    counts = screen_counts()
    assert counts["validation_evaluations"] == 66
    assert counts["benchmark_evaluations"] == 42
    assert counts["total_responses"] == 101184
    with pytest.raises(ValueError, match="complete cycles"):
        evaluation_schedule(updates=89)


def test_overlapping_suites_share_one_immutable_export_and_cannot_be_dropped(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "heartbeat.json").write_text("{}")
    checkpoint = tmp_path / "run" / "eval-snapshots" / "step_17"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".complete").touch()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"immutable")
    args = Namespace(
        opsd_eval_queue=queue, num_rollout=88, save=None,
        opsd_cyclic_teacher_policy="fixed_original", opsd_cyclic_pi_updates=7, opsd_cyclic_opd_updates=4,
    )
    enqueue_evaluation(args, 17, None, checkpoint)
    job_path = queue / "pending/run-step-0018.json"
    job = json.loads(job_path.read_text())
    assert job["evaluation_suites"] == ["validation", "benchmark"]
    job["evaluation_suites"] = ["benchmark"]
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="required suites"):
        enqueue_evaluation(args, 17, None, checkpoint)


def test_launcher_expresses_cycle_policy_without_recovery_saves(monkeypatch):
    from scripts.run_qwen3_4b_opsd_study import ScriptArgs, execute

    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("RAY_ADDRESS", "http://127.0.0.1:8265")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    calls = []
    monkeypatch.setattr(ScriptArgs, "create_backend", lambda _: SimpleNamespace(execute_train=lambda **kw: calls.append(kw)))
    execute(ScriptArgs(num_rollout=88, evaluation_queue="/queue", cyclic_teacher_policy="cycle_refresh"))
    argv = shlex.split(calls[0]["train_args"])
    assert argv[argv.index("--opsd-cyclic-teacher-policy") + 1] == "cycle_refresh"
    assert argv[argv.index("--opsd-cyclic-pi-updates") + 1] == "7"
    assert argv[argv.index("--opsd-cyclic-opd-updates") + 1] == "4"
    assert "--save" not in argv and "--no-load-optim" in argv
    assert argv[argv.index("--micro-batch-size") + 1] == "1"


@pytest.mark.parametrize("changes", [
    {"num_rollout": 87}, {"teacher_ema_decay": 0.9}, {"save_checkpoints": True},
    {"snapshot_interval": 7}, {"micro_batch_size": 2}, {"task": "code"},
])
def test_launcher_rejects_incompatible_cycle_contract(changes):
    from scripts.run_qwen3_4b_opsd_study import ScriptArgs

    config = dict(num_rollout=88, evaluation_queue="/queue", cyclic_teacher_policy="fixed_original")
    with pytest.raises(ValueError):
        ScriptArgs(**(config | changes))
