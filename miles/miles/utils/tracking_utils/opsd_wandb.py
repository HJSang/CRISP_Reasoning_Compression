"""Restricted W&B payloads for the public OPSD study.

Only explicitly named recipe fields and numeric metrics cross this boundary.
Authentication stays in the SDK's environment/netrc lookup, never in run config.
"""

import json
import math
import os
from numbers import Real
from pathlib import Path

import wandb


CONFIG_FIELDS = (
    "seed", "rollout_seed", "num_rollout", "global_batch_size", "micro_batch_size",
    "lr", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "clip_grad",
    "opsd_beta", "opsd_temperature", "opsd_token_clip", "lora_rank",
    "rollout_max_response_len", "rollout_temperature", "rollout_top_p", "rollout_top_k",
)
CONFIG_CHOICES = {
    "opsd_context": {"none", "answer", "worked", "unrelated", "empty"},
    "opsd_target": {"frozen", "current", "transported"},
    "opsd_reduction": {"sample_mean", "token_mean"},
}
METRIC_FIELDS = {
    "train/step", "train/opsd_loss", "train/opsd_valid_tokens", "train/grad_norm",
    "train/lr-pg_0", "train/lr-pg_1", "rollout/step", "rollout/response_lengths",
    "rollout/truncated", "rollout/raw_reward", "rollout/rewards", "rollout/entropy",
    "perf/actor_train_time", "perf/actor_train_tok_per_s", "perf/train_time",
    "perf/step_time", "perf/save_model_time", "perf/train_wait_time",
    "perf/update_weights_time", "perf/rollout_time", "perf/tokens_per_gpu_per_sec",
    "perf/effective_tokens_per_gpu_per_sec", "perf/longest_sample_tokens_per_sec",
}
EVAL_FIELDS = (
    "avg_at_8", "completions", "cap_hit_fraction", "parse_failure_fraction",
    "grader_timeouts", "unexpected_thinking_delimiters",
)
EVAL_DATASETS = {"AIME 2024": "AIME_2024", "AIME 2025": "AIME_2025", "AMC23": "AMC23"}


def settings(*, primary: bool, mode: str = "shared") -> wandb.Settings:
    # Explicit values override SDK/environment defaults that could reveal the host.
    return wandb.Settings(
        mode=mode, init_timeout=300, console="off", capture_loggers={}, silent=True,
        disable_code=True, disable_git=True, disable_job_creation=True,
        x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False,
        host="worker", program="opsd", program_abspath="opsd", program_relpath="opsd",
        x_label="primary" if primary else "metrics", x_primary=primary,
        x_update_finish_state=primary,
    )


def _number(value) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("OPSD tracking only accepts numeric scalar metrics")
    if not math.isfinite(value):
        raise ValueError("OPSD tracking received a non-finite metric")
    return float(value)


def config(args) -> dict:
    output = {key: _number(getattr(args, key)) for key in CONFIG_FIELDS if hasattr(args, key)}
    for key, choices in CONFIG_CHOICES.items():
        value = getattr(args, key, None)
        if value is not None:
            if value not in choices:
                raise ValueError(f"Unsupported OPSD tracking field: {key}")
            output[key] = value
    return output


def metrics(values: dict) -> dict:
    return {key: _number(value) for key, value in values.items() if key in METRIC_FIELDS}


def save_run_link(directory: Path, run) -> None:
    """The local run directory is private; never upload this operational record."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "wandb-link.tmp"
    temporary.write_text(json.dumps({"run_id": run.id, "url": run.url}))
    temporary.replace(directory / "wandb-link.json")


def evaluation_metrics(record: dict, *, completed_updates: int) -> dict:
    output = {"eval/step": completed_updates}
    for key in ("completions", "responses_per_second", "output_tokens_per_second", "mean_response_tokens"):
        output[f"eval/{key}"] = _number(record[key])
    for dataset, values in record["metrics"].items():
        if dataset not in EVAL_DATASETS:
            raise ValueError("Unexpected dataset in OPSD evaluation tracking")
        for key in EVAL_FIELDS:
            output[f"eval/{EVAL_DATASETS[dataset]}/{key}"] = _number(values[key])
    return output


def log_evaluation(identity: dict, record: dict, *, completed_updates: int) -> None:
    payload = evaluation_metrics(record, completed_updates=completed_updates)
    # The queue can inherit the trainer's SDK service through their common parent.
    # A shared hosted run still needs a separate local service in each process.
    os.environ.pop("WANDB_SERVICE", None)
    # A secondary writer must not finish the still-running trainer's shared run.
    with wandb.init(
        entity=identity["entity"], project=identity["project"], id=identity["run_id"],
        resume="allow", settings=settings(primary=False), config={},
    ) as run:
        run.log(payload)
