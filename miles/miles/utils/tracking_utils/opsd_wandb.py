"""Restricted W&B payloads for the public OPSD study.

Only explicitly named recipe fields and numeric metrics cross this boundary.
Authentication stays in the SDK's environment/netrc lookup, never in run config.
"""

import json
import math
import os
import re
from numbers import Real
from pathlib import Path

import wandb


CONFIG_FIELDS = (
    "seed", "rollout_seed", "num_rollout", "global_batch_size", "micro_batch_size",
    "lr", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "clip_grad",
    "opsd_beta", "opsd_temperature", "opsd_token_clip", "lora_rank",
    "rollout_max_response_len", "rollout_temperature", "rollout_top_p", "rollout_top_k", "opsd_teacher_ema_decay",
    "opsd_cyclic_pi_updates", "opsd_cyclic_opd_updates",
)
CONFIG_CHOICES = {
    "opsd_context": {"none", "answer", "worked", "unrelated", "empty"},
    "opsd_target": {"frozen", "current", "transported"},
    "opsd_reduction": {"sample_mean", "token_mean"},
    "opsd_task": {"math", "code"},
    "opsd_cyclic_teacher_policy": {"fixed_original", "cycle_refresh"},
}
METRIC_FIELDS = {
    "train/step", "train/opsd_loss", "train/opsd_valid_tokens", "train/grad_norm",
    "train/lr-pg_0", "train/lr-pg_1", "rollout/step", "rollout/response_lengths",
    "rollout/truncated", "rollout/raw_reward", "rollout/rewards", "rollout/entropy",
    "perf/actor_train_time", "perf/actor_train_tok_per_s", "perf/train_time",
    "perf/step_time", "perf/save_model_time", "perf/train_wait_time",
    "perf/update_weights_time", "perf/rollout_time", "perf/tokens_per_gpu_per_sec",
    "perf/effective_tokens_per_gpu_per_sec", "perf/longest_sample_tokens_per_sec",
    "train/ema_updates", "train/ema_local_student_distance", "train/ema_local_original_distance",
    "train/cyclic_cycle", "train/cyclic_pi_phase", "train/cyclic_optimizer_reset",
    "train/cyclic_teacher_source_update", "train/cyclic_teacher_refreshed", "train/cyclic_completed_updates",
}
EVAL_FIELDS = (
    "avg_at_8", "completions", "cap_hit_fraction", "parse_failure_fraction",
    "grader_timeouts", "unexpected_thinking_delimiters",
)
EVAL_DATASETS = {"AIME 2024": "AIME_2024", "AIME 2025": "AIME_2025", "AMC23": "AMC23", "HumanEval+": "HumanEval_plus"}
VALIDATION_DATASETS = {"Math validation": "Math_validation"}


def run_name(
    *, target: str, context: str, seed: int, group: str, task: str = "math", ema_decay: float | None = None,
    cyclic_teacher_policy: str | None = None, cyclic_pi_updates: int | None = None,
    cyclic_opd_updates: int | None = None, planned_updates: int | None = None,
) -> str:
    """Describe the recipe without exposing arbitrary group names or local paths."""
    cyclic = re.fullmatch(r"mathcycles-qwen3-(4b|8b)-(fixed|refresh)-s([0-9]+)-r([0-9]+)", group)
    if cyclic_teacher_policy is not None or group.startswith("mathcycles-"):
        if cyclic is None:
            raise ValueError("Cyclic study requires a recognized neutral run identifier")
        size, policy, run_seed, attempt = cyclic.groups()
        expected_policy = {"fixed": "fixed_original", "refresh": "cycle_refresh"}[policy]
        if (int(run_seed) != seed or target != "frozen" or context != "worked" or task != "math"
                or ema_decay is not None or cyclic_teacher_policy != expected_policy):
            raise ValueError("Cyclic run name disagrees with the recipe or teacher policy")
        counts = (cyclic_pi_updates, cyclic_opd_updates, planned_updates)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts):
            raise ValueError("Cyclic run name requires positive integer phase and total updates")
        cycle_updates = cyclic_pi_updates + cyclic_opd_updates
        if planned_updates % cycle_updates:
            raise ValueError("Cyclic run name requires complete PI/OPD cycles")
        teacher = "fixed original teacher" if policy == "fixed" else "cycle-refreshed teacher"
        return (f"Qwen3-{size.upper()} | math | {teacher} | PI{cyclic_pi_updates} → OPD{cyclic_opd_updates}"
                f" × {planned_updates // cycle_updates} | fresh Adam per phase | seed {seed}"
                f" | non-thinking | attempt {int(attempt)}")
    cell = re.fullmatch(r"generality-qwen3-(1p7b|8b)-(math|code)-(frozen|ema)-(warmup|pi|opd)-s([0-9]+)-r([0-9]+)", group)
    if cell:
        size, task_name, policy, phase, run_seed, attempt = cell.groups()
        expected_ema = policy == "ema" and phase != "opd"
        if int(run_seed) != seed or target != "frozen" or task_name != task:
            raise ValueError("Generality run name disagrees with the recipe")
        if context != ("none" if phase == "opd" else "worked") or expected_ema != (ema_decay is not None):
            raise ValueError("Generality run name disagrees with the teacher policy")
        model = {"1p7b": "Qwen3-1.7B", "8b": "Qwen3-8B"}[size]
        warmup = "EMA PI7" if policy == "ema" else "frozen PI7"
        label = warmup if phase == "warmup" else warmup + (" → original OPD4" if phase == "opd" else " → continued PI4")
        return f"{model} | {task} | {label} | seed {seed} | non-thinking | attempt {int(attempt)}"
    replicate = re.fullmatch(r"replication-s([0-9]+)-(warmup|opd|pi)", group)
    if replicate:
        if int(replicate[1]) != seed or target != "frozen":
            raise ValueError("Replication name disagrees with the seed or teacher")
        phase = replicate[2]
        expected_context = "none" if phase == "opd" else "worked"
        if context != expected_context:
            raise ValueError("Replication name disagrees with the PI context")
        label = {"warmup": "PI warm-up 7", "opd": "PI7 → original-teacher OPD4", "pi": "PI7 → continued PI4"}[phase]
        return f"Qwen3-4B | {label} | replicate {seed} | non-thinking"
    teacher = "EMA teacher" if ema_decay is not None else "current teacher" if target == "current" else "frozen teacher"
    information = {
        "none": "no PI", "answer": "answer PI", "worked": "worked-solution PI",
        "unrelated": "unrelated PI", "empty": "empty PI wrapper",
    }[context]
    target_label = {"frozen": "direct target", "current": "direct target", "transported": "transported target"}[target]
    parts = [teacher, information, target_label]
    # Only the known neutral study identifier contributes block/attempt metadata.
    branch = re.fullmatch(r"B(?:01|10|11|S)(?:-r(?P<attempt>[0-9]+))?-block-(?P<block>[0-9]+)", group)
    if branch:
        parts.append(f"block {int(branch['block']):02d}")
    elif re.fullmatch(r"B-warmup-u[0-9]+", group):
        parts.append("warm-up")
    parts.append(f"seed {seed}")
    if branch and branch["attempt"] is not None:
        parts.append(f"attempt {int(branch['attempt']):02d}")
    return " | ".join(parts)


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
    output = {key: _number(getattr(args, key)) for key in CONFIG_FIELDS if getattr(args, key, None) is not None}
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


def evaluation_metrics(record: dict, *, completed_updates: int, suite: str = "benchmark") -> dict:
    if suite not in {"benchmark", "validation"}:
        raise ValueError("Unexpected OPSD evaluation suite")
    prefix = "eval" if suite == "benchmark" else "validation"
    datasets = EVAL_DATASETS if suite == "benchmark" else VALIDATION_DATASETS
    output = {f"{prefix}/step": completed_updates}
    for key in ("completions", "responses_per_second", "output_tokens_per_second", "mean_response_tokens"):
        output[f"{prefix}/{key}"] = _number(record[key])
    for dataset, values in record["metrics"].items():
        if dataset not in datasets:
            raise ValueError("Unexpected dataset in OPSD evaluation tracking")
        for key in EVAL_FIELDS:
            output[f"{prefix}/{datasets[dataset]}/{key}"] = _number(values[key])
        for key in ("pass_at_8", "test_failure_fraction"):
            if key in values:
                output[f"{prefix}/{datasets[dataset]}/{key}"] = _number(values[key])
    return output


def log_evaluation(identity: dict, record: dict, *, completed_updates: int, suite: str = "benchmark") -> None:
    payload = evaluation_metrics(record, completed_updates=completed_updates, suite=suite)
    # The queue can inherit the trainer's SDK service through their common parent.
    # A shared hosted run still needs a separate local service in each process.
    os.environ.pop("WANDB_SERVICE", None)
    # A secondary writer must not finish the still-running trainer's shared run.
    with wandb.init(
        entity=identity["entity"], project=identity["project"], id=identity["run_id"],
        resume="allow", settings=settings(primary=False), config={},
    ) as run:
        run.log(payload)
