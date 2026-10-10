"""Evaluate immutable study snapshots on an already running, isolated SGLang fleet.

Run one worker per queue. Failures are persisted for the save hook to stop the
training job; no evaluation point is silently skipped. All outputs are private.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from miles.utils.opsd_checkpoint import checkpoint_digest
from miles.utils.opsd_study import retire_evaluation_snapshot
from miles.utils.tracking_utils.opsd_wandb import log_evaluation


def _write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def _heartbeat(queue, stop):
    while not stop.is_set():
        _write(queue / "heartbeat.json", {"unix_time": time.time()})
        stop.wait(5)


def _pin_protocol(args):
    """Refuse queue reuse with a different seed bank, recipe, or benchmark tape."""
    protocol = {"seed_namespace": args.seed_namespace}
    for name in ("plan", "prompts", "labels"):
        protocol[name + "_sha256"] = hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
    validation = [getattr(args, "validation_" + name, None) for name in ("plan", "prompts", "labels")]
    if any(path is not None for path in validation):
        if any(path is None for path in validation):
            raise ValueError("Validation requires plan, prompts and labels together")
        for name, path in zip(("plan", "prompts", "labels"), validation, strict=True):
            protocol["validation_" + name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = args.queue / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Evaluation queue protocol changed; use a fresh queue")
    _write(path, protocol)
    return protocol


@dataclass(frozen=True)
class _EvaluationSuite:
    plan: Path
    prompts: Path
    labels: Path
    expected_completions: int


def _evaluation_suites(args):
    suites = {}
    for name, prefix in (("benchmark", ""), ("validation", "validation_")):
        plan = getattr(args, prefix + "plan", None)
        if plan is None:
            continue
        prompts, labels = getattr(args, prefix + "prompts"), getattr(args, prefix + "labels")
        spec = json.loads(plan.read_text())["evaluation"]
        suites[name] = _EvaluationSuite(
            plan, prompts, labels, len(prompts.read_text().splitlines()) * spec["samples_per_question"],
        )
    return suites


def _requested_suites(job, available):
    names = job.get("evaluation_suites", ["benchmark"])
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("Evaluation suites must be a nonempty list of configured suite names")
    if len(set(names)) != len(names) or not set(names) <= set(available):
        raise ValueError("Evaluation suites contain duplicates or an unconfigured suite")
    return names


def _evaluate_suite(args, active, job, name, suite, protocol):
    suffix = "" if name == "benchmark" else ".validation"
    result = args.queue / "results" / f"{active.stem}{suffix}.json"
    command = [
        sys.executable, str(Path(__file__).with_name("evaluate_checkpoint.py")),
        "--model", args.model, "--plan", str(suite.plan), "--prompts", str(suite.prompts),
        "--labels", str(suite.labels), "--checkpoint", job["checkpoint"], "--output", str(result),
        "--concurrency", str(args.concurrency), "--seed-namespace", args.seed_namespace, "--urls", *args.urls,
    ]
    remaining = args.deadline_unix - time.time()
    if remaining <= 0:
        raise TimeoutError("Evaluation deadline reached before the next suite")
    with (args.queue / "logs" / f"{active.stem}{suffix}.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=min(3600, remaining))
    if completed.returncode != 0:
        raise RuntimeError(f"{name} evaluation process exited {completed.returncode}")
    record = json.loads(result.read_text())
    if record["status"] != "completed":
        raise ValueError("Evaluation did not produce a completed result")
    if record["provenance"]["seed_namespace"] != args.seed_namespace:
        raise ValueError("Evaluation seed bank mismatch")
    if record["checkpoint_sha256"] != job["checkpoint_sha256"] or record["completions"] != suite.expected_completions:
        raise ValueError("Evaluation hash or completion-count mismatch")
    prefix = "" if name == "benchmark" else "validation_"
    for asset in ("prompts", "labels"):
        if record[asset + "_sha256"] != protocol[prefix + asset + "_sha256"]:
            raise ValueError("Evaluation input hash differs from the pinned suite")
    if identity := job.get("wandb"):
        log_evaluation(identity, record, completed_updates=job["completed_updates"], suite=name)
    return str(result)


def _complete_job(args, active, job, suites, protocol):
    names = _requested_suites(job, suites)
    if job.get("checkpoint_kind") != "full_model":
        raise ValueError("Study queue requires full-model checkpoints; use a fresh queue after LoRA")
    results = {}
    for name in names:
        # Each suite reloads this same immutable export; failed tracking retains it.
        if _pin_protocol(args) != protocol or checkpoint_digest(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
            raise ValueError("Evaluation source changed after enqueue")
        results[name] = _evaluate_suite(args, active, job, name, suites[name], protocol)
    retire_evaluation_snapshot(job)
    _write(args.queue / "done" / active.name, job | {
        "result": results.get("benchmark", next(iter(results.values()))),
        "results": results, "exit_code": 0, "protocol": protocol,
    })
    active.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path)
    parser.add_argument("--validation-prompts", type=Path)
    parser.add_argument("--validation-labels", type=Path)
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--seed-namespace", default="opsd-eval-17")
    args = parser.parse_args()
    for name in ["pending", "running", "done", "failed", "results", "logs"]:
        (args.queue / name).mkdir(parents=True, exist_ok=True)
    protocol = _pin_protocol(args)
    suites = _evaluation_suites(args)
    if list((args.queue / "running").glob("*.json")):
        raise RuntimeError("Unresolved running jobs need inspection before restarting the evaluator")
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(args.queue, stop), daemon=True)
    heartbeat.start()
    try:
        while time.time() < args.deadline_unix:
            if list((args.queue / "failed").glob("*.json")):
                raise RuntimeError("A prior evaluation failed; inspect it before continuing")
            pending = sorted((args.queue / "pending").glob("*.json"))
            if not pending:
                time.sleep(1)
                continue
            active = args.queue / "running" / pending[0].name
            pending[0].replace(active)
            job = json.loads(active.read_text())
            try:
                _complete_job(args, active, job, suites, protocol)
            except Exception as error:
                _write(args.queue / "failed" / active.name, job | {"exit_code": 1, "error": str(error)})
                raise
    finally:
        stop.set()
        heartbeat.join(timeout=6)
        (args.queue / "heartbeat.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
