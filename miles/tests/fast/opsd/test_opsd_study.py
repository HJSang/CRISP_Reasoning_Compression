"""Checkpoint cadence, queue failure handling and metric invariance."""

import json
from argparse import Namespace

import pytest
import torch
from tests.fast.opsd.test_opsd_contract import _loss_fixture

from miles.backends.training_utils.loss.hub.opsd import opsd_loss_function
from miles.utils.opsd_checkpoint import checkpoint_digest
from miles.utils.opsd_study import enqueue_evaluation, evaluation_steps


@pytest.mark.parametrize("missing", ["MILES_SCRIPT_EXTERNAL_RAY", "RAY_ADDRESS"])
def test_study_launcher_requires_explicit_cluster_before_any_commands(monkeypatch, missing):
    from scripts.run_qwen3_4b_opsd_study import execute

    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("RAY_ADDRESS", "http://127.0.0.1:8265")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="isolated Ray cluster"):
        execute(None)


def test_percentage_cadence_deduplicates_and_includes_final_once():
    assert evaluation_steps(4) == (1, 2, 3, 4)
    assert evaluation_steps(64) == (7, 13, 20, 26, 32, 39, 45, 52, 58, 64)
    assert evaluation_steps(1) == (1,)
    with pytest.raises(ValueError):
        evaluation_steps(0)


@pytest.mark.parametrize("stop", [0, 8])
def test_study_early_stop_must_fit_planned_updates(stop):
    from scripts.run_qwen3_4b_opsd_study import ScriptArgs

    with pytest.raises(ValueError, match="Early stopping"):
        ScriptArgs(num_rollout=7, stop_after_rollout=stop)


@pytest.mark.parametrize("stop", [5, 7])
def test_early_final_checkpoint_is_evaluated_and_drained(tmp_path, monkeypatch, stop):
    import miles.utils.opsd_study as study

    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "heartbeat.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".complete").touch()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"immutable-model")
    args = Namespace(
        opsd_eval_queue=queue, num_rollout=64, save=str(tmp_path / "run/checkpoints"),
        debug_exit_after_rollout=stop, start_rollout_id=0,
    )
    completed = []

    def finish_pending(_):
        pending = list((queue / "pending").glob("*.json"))
        assert len(pending) == 1
        job = pending[0]
        completed.append(json.loads(job.read_text())["completed_updates"])
        job.replace(queue / "done" / job.name)

    monkeypatch.setattr(study.time, "sleep", finish_pending)
    enqueue_evaluation(args, stop - 1, checkpoint, checkpoint)
    assert completed == [stop]
    assert not list((queue / "pending").glob("*.json"))


def test_queue_uses_completed_updates_and_refuses_changed_checkpoint(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "heartbeat.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".complete").touch()
    (checkpoint / "config.json").write_text('{"model_type":"qwen3"}')
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"immutable-test-model")
    args = Namespace(opsd_eval_queue=queue, num_rollout=64, save=str(tmp_path / "run/checkpoints"))
    enqueue_evaluation(args, 5, checkpoint, None)
    assert not (queue / "pending").exists()
    with pytest.raises(RuntimeError, match="--save-hf"):
        enqueue_evaluation(args, 6, checkpoint, None)
    enqueue_evaluation(args, 6, checkpoint, checkpoint)
    job = queue / "pending/run-step-0007.json"
    assert json.loads(job.read_text())["completed_updates"] == 7
    assert json.loads(job.read_text())["checkpoint_kind"] == "full_model"
    enqueue_evaluation(args, 6, checkpoint, checkpoint)
    assert len(list((queue / "pending").glob("*.json"))) == 1
    weights.write_bytes(b"modified-model")
    with pytest.raises(ValueError, match="collides"):
        enqueue_evaluation(args, 6, checkpoint, checkpoint)
    (queue / "failed/bad.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        enqueue_evaluation(args, 6, checkpoint, checkpoint)


def test_full_checkpoint_identity_covers_metadata_and_rejects_partial_exports(tmp_path):
    with pytest.raises(ValueError, match="completion marker"):
        checkpoint_digest(tmp_path)
    (tmp_path / ".complete").touch()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    original = checkpoint_digest(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template":"new-template"}')
    assert checkpoint_digest(tmp_path) != original
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{"weight":"missing.safetensors"}}')
    with pytest.raises(ValueError, match="missing an indexed shard"):
        checkpoint_digest(tmp_path)


def test_logged_loss_matches_sample_mean_not_inverse_microbatch(monkeypatch):
    from miles.backends.training_utils import parallel

    monkeypatch.setattr(parallel, "_parallel_state", Namespace(tp=Namespace(rank=0, size=1, group=None)))
    logits, batch, _ = _loss_fixture()
    args = Namespace(opsd_beta=0, opsd_temperature=1, opsd_token_clip=0, vocab_size=5, opsd_reduction="sample_mean")
    loss, _, log = opsd_loss_function(args, batch, logits)
    torch.testing.assert_close(log["values"][1] / 2, loss)
    assert log["values"][0] == 2


def test_full_update_capture_cannot_run_on_live_study(monkeypatch):
    from scripts.run_qwen3_4b_opsd_study import ScriptArgs, execute

    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("RAY_ADDRESS", "http://127.0.0.1:8265")
    with pytest.raises(ValueError, match="correctness replay"):
        execute(ScriptArgs(capture_full_update=True))
