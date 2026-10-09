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
    path = args.queue / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Evaluation queue protocol changed; use a fresh queue")
    _write(path, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--seed-namespace", default="opsd-eval-17")
    args = parser.parse_args()
    for name in ["pending", "running", "done", "failed", "results", "logs"]:
        (args.queue / name).mkdir(parents=True, exist_ok=True)
    protocol = _pin_protocol(args)
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
            result = args.queue / "results" / active.name
            code = 1
            try:
                if job.get("checkpoint_kind") != "full_model":
                    raise ValueError("Study queue requires full-model checkpoints; use a fresh queue after LoRA")
                checkpoint = Path(job["checkpoint"])
                if checkpoint_digest(checkpoint) != job["checkpoint_sha256"]:
                    raise ValueError("Checkpoint hash changed after enqueue")
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("evaluate_checkpoint.py")),
                    "--model",
                    args.model,
                    "--plan",
                    str(args.plan),
                    "--prompts",
                    str(args.prompts),
                    "--labels",
                    str(args.labels),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(result),
                    "--concurrency",
                    str(args.concurrency),
                    "--seed-namespace",
                    args.seed_namespace,
                    "--urls",
                    *args.urls,
                ]
                with (args.queue / "logs" / active.with_suffix(".log").name).open("w") as log:
                    completed = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=min(3600, max(1, args.deadline_unix - time.time())),
                    )
                code = completed.returncode
                if code != 0:
                    raise RuntimeError(f"Evaluation process exited {code}")
                record = json.loads(result.read_text())
                if record["provenance"]["seed_namespace"] != args.seed_namespace:
                    raise ValueError("Evaluation seed bank mismatch")
                if record["checkpoint_sha256"] != job["checkpoint_sha256"] or record["completions"] != 800:
                    raise ValueError("Evaluation hash or completion-count mismatch")
                if identity := job.get("wandb"):
                    log_evaluation(identity, record, completed_updates=job["completed_updates"])
                retire_evaluation_snapshot(job)
                _write(args.queue / "done" / active.name, job | {"result": str(result), "exit_code": code, "protocol": protocol})
                active.unlink()
            except Exception as error:
                _write(args.queue / "failed" / active.name, job | {"exit_code": code, "error": str(error)})
                raise
    finally:
        stop.set()
        heartbeat.join(timeout=6)
        (args.queue / "heartbeat.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
