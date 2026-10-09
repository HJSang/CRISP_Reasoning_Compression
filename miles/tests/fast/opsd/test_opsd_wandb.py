"""Protect the study's public tracking boundary, including secondary writers."""

import json
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.common import get_default_wandb_args
from miles.utils.tracking_utils import opsd_wandb, wandb_utils
from miles.utils.tracking_utils.base import WandbBackend


def test_config_and_metrics_exclude_identity_paths_and_text():
    args = Namespace(
        seed=17, lr=5e-6, opsd_target="transported", opsd_context="worked",
        wandb_key="test-credential", load="/private/model", host="private-worker",
        env_report="/private/launch.json", prompt="private training example",
    )
    assert opsd_wandb.config(args) == {
        "seed": 17.0, "lr": 5e-6, "opsd_target": "transported", "opsd_context": "worked",
    }
    metrics = opsd_wandb.metrics({"train/opsd_loss": 0.12, "train/private-worker": 4, "sample": "private"})
    assert metrics == {"train/opsd_loss": 0.12}
    with pytest.raises(ValueError, match="non-finite"):
        opsd_wandb.metrics({"train/opsd_loss": float("nan")})
    with pytest.raises(TypeError, match="numeric scalar"):
        opsd_wandb.metrics({"train/opsd_loss": "private"})


@pytest.mark.parametrize("primary", [True, False])
def test_all_writers_disable_automatic_capture(primary):
    settings = opsd_wandb.settings(primary=primary)
    assert settings.console == "off" and settings.capture_loggers == {}
    assert settings.disable_code and settings.disable_git and settings.disable_job_creation
    assert settings.x_disable_meta and settings.x_disable_stats and settings.x_disable_machine_info
    assert settings.x_save_requirements is False
    assert settings.host == "worker" and settings.program == "opsd"
    assert settings.x_primary is primary and settings.x_update_finish_state is primary


@pytest.mark.parametrize("primary", [True, False])
def test_existing_wandb_init_uses_profile_for_every_writer(monkeypatch, primary):
    calls = []
    monkeypatch.setenv("WANDB_SERVICE", "inherited-primary-service")
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kw: calls.append(kw))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *a, **kw: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="test-run"))
    args = Namespace(
        use_wandb=True, wandb_opsd_profile=True, wandb_mode="online", wandb_key=None,
        wandb_random_suffix=False, wandb_group="B10-block-00", wandb_team="team",
        wandb_project="opsd", wandb_run_id="test-run", wandb_dir=None,
        sglang_enable_metrics=True, env_report="/private/launch.json", seed=17, save=None, save_hf=None,
        opsd_target="transported", opsd_context="worked",
    )
    if primary:
        wandb_utils.init_wandb_primary(args)
    else:
        wandb_utils.init_wandb_secondary(args, router_addr="http://private-router")
        assert "WANDB_SERVICE" not in wandb_utils.os.environ
    assert calls[0]["config"] == {"seed": 17.0, "opsd_target": "transported", "opsd_context": "worked"}
    assert calls[0]["settings"].x_disable_stats
    assert calls[0]["settings"].x_stats_open_metrics_endpoints is None


def test_framework_backend_filters_actual_log_calls(monkeypatch):
    logged = []
    monkeypatch.setattr(opsd_wandb.wandb, "log", logged.append)
    monkeypatch.setattr(wandb_utils, "init_wandb_secondary", lambda *a, **kw: None)
    backend = WandbBackend()
    backend.init(Namespace(wandb_opsd_profile=True), primary=False)
    backend.log({"train/step": 0, "train/opsd_loss": 0.2, "response": "private"})
    backend.log({"host": "private-worker"})
    assert logged == [{"train/step": 0.0, "train/opsd_loss": 0.2}]


def test_launcher_uses_sdk_auth_without_credentials_in_argv(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-credential")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "opsd")
    argv = get_default_wandb_args(__file__, run_id="B10-block-00", opsd_profile=True)
    assert "--wandb-opsd-profile" in argv and "--wandb-team team" in argv
    assert "--wandb-key" not in argv and "test-credential" not in argv
    monkeypatch.delenv("WANDB_ENTITY")
    with pytest.raises(ValueError, match="WANDB_ENTITY"):
        get_default_wandb_args(__file__, run_id="B10-block-00", opsd_profile=True)


def test_run_link_is_saved_as_a_local_operational_record(tmp_path):
    opsd_wandb.save_run_link(tmp_path, SimpleNamespace(id="test-run", url="https://wandb.ai/team/opsd/runs/test-run"))
    assert json.loads((tmp_path / "wandb-link.json").read_text())["run_id"] == "test-run"
    assert not (tmp_path / "wandb-link.tmp").exists()


def test_evaluation_payload_keeps_scores_but_excludes_responses_and_provenance():
    record = {
        "completions": 800, "responses_per_second": 3.2,
        "output_tokens_per_second": 8000, "mean_response_tokens": 2500,
        "metrics": {name: dict.fromkeys(opsd_wandb.EVAL_FIELDS, 0.1) for name in opsd_wandb.EVAL_DATASETS},
        "responses": [{"text": "private example"}], "provenance": {"host": "private-worker"},
    }
    payload = opsd_wandb.evaluation_metrics(record, completed_updates=4)
    assert payload["eval/step"] == 4 and payload["eval/AIME_2024/avg_at_8"] == 0.1
    assert "private" not in json.dumps(payload)
    record["metrics"]["private-worker"] = {}
    with pytest.raises(ValueError, match="Unexpected dataset"):
        opsd_wandb.evaluation_metrics(record, completed_updates=4)


def test_evaluation_writer_joins_without_finishing_the_training_run(monkeypatch):
    calls, logged = [], []
    monkeypatch.setenv("WANDB_SERVICE", "inherited-primary-service")

    def init(**kwargs):
        assert "WANDB_SERVICE" not in opsd_wandb.os.environ
        calls.append(kwargs)
        return nullcontext(SimpleNamespace(log=logged.append))

    monkeypatch.setattr(opsd_wandb.wandb, "init", init)
    record = dict(completions=800, responses_per_second=3, output_tokens_per_second=9000,
                  mean_response_tokens=3000, metrics={})
    identity = dict(entity="team", project="opsd", run_id="test-run")
    opsd_wandb.log_evaluation(identity, record, completed_updates=4)
    assert calls[0]["id"] == "test-run" and calls[0]["config"] == {}
    assert calls[0]["settings"].x_update_finish_state is False
    assert logged[0]["eval/step"] == 4
