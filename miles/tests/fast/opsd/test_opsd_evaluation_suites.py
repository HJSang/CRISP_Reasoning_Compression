"""A phase snapshot survives until every declared evaluation suite is tracked."""

import hashlib
import json
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.opsd_checkpoint import checkpoint_digest
from tools.opsd import serve_evaluation_queue as queue


@pytest.fixture
def evaluation_job(tmp_path):
    for name in ("running", "done", "results", "logs"):
        (tmp_path / name).mkdir()
    assets = {}
    for prefix in ("", "validation_"):
        for asset, contents in (("plan", '{"evaluation":{"samples_per_question":8}}'),
                                ("prompts", '{"id":"question"}\n'), ("labels", '{"answer":"1"}\n')):
            path = tmp_path / (prefix + asset)
            path.write_text(contents)
            assets[prefix + asset] = path
    args = Namespace(queue=tmp_path, model="model", concurrency=32, urls=["http://localhost:31000"],
                     deadline_unix=time.time() + 60, seed_namespace="cycle-test", **assets)
    checkpoint = tmp_path / "eval-snapshots" / "step_17"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".complete").touch()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"immutable weights")
    job = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_digest(checkpoint),
               checkpoint_kind="full_model", completed_updates=18, rollout_id=17, temporary_snapshot=True,
               evaluation_suites=["benchmark", "validation"], wandb={"run_id": "cycle-test"})
    active = tmp_path / "running" / "cycle-step-0018.json"
    active.write_text(json.dumps(job))
    return args, active, job


def _fake_evaluator(command, **kwargs):
    value = lambda flag: command[command.index(flag) + 1]
    output = Path(value("--output"))
    output.write_text(json.dumps({
        "status": "completed", "checkpoint_sha256": checkpoint_digest(Path(value("--checkpoint"))),
        "completions": 8, "provenance": {"seed_namespace": value("--seed-namespace")},
        "prompts_sha256": hashlib.sha256(Path(value("--prompts")).read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(Path(value("--labels")).read_bytes()).hexdigest(),
    }))
    return SimpleNamespace(returncode=0)


def test_validation_protocol_requires_all_assets_and_pins_them(evaluation_job):
    args, _, _ = evaluation_job
    original = queue._pin_protocol(args)
    assert "plan_sha256" in original and "validation_plan_sha256" in original
    args.validation_prompts.write_text('{"id":"changed"}\n')
    with pytest.raises(ValueError, match="protocol changed"):
        queue._pin_protocol(args)
    args.validation_labels = None
    with pytest.raises(ValueError, match="together"):
        queue._pin_protocol(args)


@pytest.mark.parametrize("names", [[], ["validation", "validation"], ["other"], "validation", [None]])
def test_invalid_suite_requests_are_rejected(names):
    with pytest.raises(ValueError, match="suite"):
        queue._requested_suites({"evaluation_suites": names}, {"benchmark", "validation"})
    with pytest.raises(ValueError, match="unconfigured"):
        queue._requested_suites({"evaluation_suites": ["validation"]}, {"benchmark"})


def test_legacy_jobs_keep_the_benchmark_default():
    assert queue._requested_suites({}, {"benchmark"}) == ["benchmark"]


def test_overlapping_suites_share_source_and_retire_after_both_tracked(evaluation_job, monkeypatch):
    args, active, job = evaluation_job
    tracked = []
    monkeypatch.setattr(queue.subprocess, "run", _fake_evaluator)

    def track(identity, record, *, completed_updates, suite):
        assert Path(job["checkpoint"]).is_dir() and active.exists()
        assert record["checkpoint_sha256"] == job["checkpoint_sha256"] and completed_updates == 18
        tracked.append(suite)

    monkeypatch.setattr(queue, "log_evaluation", track)
    queue._complete_job(args, active, job, queue._evaluation_suites(args), queue._pin_protocol(args))
    done = json.loads((args.queue / "done" / active.name).read_text())
    assert tracked == ["benchmark", "validation"]
    assert Path(done["result"]).name == active.name
    assert Path(done["results"]["validation"]).name == "cycle-step-0018.validation.json"
    assert not Path(job["checkpoint"]).exists() and not active.exists()


@pytest.mark.parametrize("failure", ["generation", "tracking", "input_hash"])
def test_second_suite_failure_preserves_snapshot_and_earlier_result(evaluation_job, monkeypatch, failure):
    args, active, job = evaluation_job

    def run(command, **kwargs):
        validation = str(args.validation_plan) in command
        if validation and failure == "generation":
            return SimpleNamespace(returncode=7)
        result = _fake_evaluator(command, **kwargs)
        if validation and failure == "input_hash":
            path = Path(command[command.index("--output") + 1])
            record = json.loads(path.read_text()) | {"labels_sha256": "wrong labels"}
            path.write_text(json.dumps(record))
        return result

    def track(identity, record, *, completed_updates, suite):
        if suite == "validation" and failure == "tracking":
            raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(queue.subprocess, "run", run)
    monkeypatch.setattr(queue, "log_evaluation", track)
    with pytest.raises((RuntimeError, ValueError)):
        queue._complete_job(args, active, job, queue._evaluation_suites(args), queue._pin_protocol(args))
    assert Path(job["checkpoint"]).is_dir() and active.exists()
    assert (args.queue / "results" / active.name).exists()
    assert not (args.queue / "done" / active.name).exists()


def test_validation_only_phase_keeps_a_distinct_result(evaluation_job, monkeypatch):
    args, active, job = evaluation_job
    job["evaluation_suites"] = ["validation"]
    monkeypatch.setattr(queue.subprocess, "run", _fake_evaluator)
    monkeypatch.setattr(queue, "log_evaluation", lambda *a, **kw: None)
    queue._complete_job(args, active, job, queue._evaluation_suites(args), queue._pin_protocol(args))
    done = json.loads((args.queue / "done" / active.name).read_text())
    assert set(done["results"]) == {"validation"}
    assert done["result"] == done["results"]["validation"]
