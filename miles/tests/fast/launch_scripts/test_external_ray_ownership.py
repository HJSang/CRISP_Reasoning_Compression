"""A launcher must not kill services owned by an external Ray cluster."""

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from tests.fast.launch_scripts.py_harness import freeze_environment
from tests.fast.utils.command_recorder import record_commands


def test_external_ray_launch_never_runs_global_cleanup(monkeypatch):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("RAY_ADDRESS", "http://127.0.0.1:8275")
    commands = record_commands(monkeypatch)
    ExecuteTrainConfig().create_backend().execute_train(
        train_args="--train-backend megatron",
        num_gpus_per_node=2,
        megatron_model_type="qwen3-4B",
        job_lifetime="launcher",
    )
    assert any("ray job submit" in command and "8275" in command for command in commands)
    assert all(
        "pkill" not in command and "ray stop" not in command and "ray start" not in command for command in commands
    )
