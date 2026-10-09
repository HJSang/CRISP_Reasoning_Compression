"""Checkpoint cadence, queue failure handling and metric invariance."""

import json
from argparse import Namespace

import pytest
import torch
from tests.fast.opsd.test_opsd_contract import _loss_fixture

from miles.backends.training_utils.loss.hub.opsd import opsd_loss_function
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


def test_queue_uses_completed_updates_and_refuses_changed_checkpoint(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "heartbeat.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "adapter").mkdir(parents=True)
    weights = checkpoint / "adapter/adapter_model.safetensors"
    weights.write_bytes(b"immutable-test-adapter")
    args = Namespace(opsd_eval_queue=queue, num_rollout=64, save=str(tmp_path / "run/checkpoints"))
    enqueue_evaluation(args, 5, checkpoint, None)
    assert not (queue / "pending").exists()
    enqueue_evaluation(args, 6, checkpoint, None)
    job = queue / "pending/run-step-0007.json"
    assert json.loads(job.read_text())["completed_updates"] == 7
    enqueue_evaluation(args, 6, checkpoint, None)
    assert len(list((queue / "pending").glob("*.json"))) == 1
    weights.write_bytes(b"modified-adapter")
    with pytest.raises(ValueError, match="collides"):
        enqueue_evaluation(args, 6, checkpoint, None)
    (queue / "failed/bad.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        enqueue_evaluation(args, 6, checkpoint, None)


def test_logged_loss_matches_sample_mean_not_inverse_microbatch(monkeypatch):
    from miles.backends.training_utils import parallel

    monkeypatch.setattr(parallel, "_parallel_state", Namespace(tp=Namespace(rank=0, size=1, group=None)))
    logits, batch, _ = _loss_fixture()
    args = Namespace(opsd_beta=0, opsd_temperature=1, opsd_token_clip=0, vocab_size=5, opsd_reduction="sample_mean")
    loss, _, log = opsd_loss_function(args, batch, logits)
    torch.testing.assert_close(log["values"][1] / 2, loss)
    assert log["values"][0] == 2
