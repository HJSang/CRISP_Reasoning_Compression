"""Study checkpoint evaluation cadence and bounded queue, outside the trainer."""

import json
import shutil
import time
from pathlib import Path

from miles.utils.opsd_checkpoint import checkpoint_digest


def evaluation_steps(updates: int) -> tuple[int, ...]:
    if updates <= 0:
        raise ValueError("Planned updates must be positive")
    return tuple(sorted({(k * updates + 9) // 10 for k in range(1, 11)}))


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _check_worker(queue: Path) -> None:
    failures = list((queue / "failed").glob("*.json"))
    if failures:
        raise RuntimeError(f"Evaluation failed; inspect {failures[0]}")
    heartbeat = queue / "heartbeat.json"
    if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 120:
        raise RuntimeError("Evaluation worker heartbeat is absent or stale; training must pause")


def enqueue_evaluation(args, rollout_id, checkpoint_dir, hf_checkpoint_dir):
    """Run after a completed HF export; final update drains this run.

    Miles checkpoint directory indices are zero-based rollout IDs. Jobs expose
    the completed optimizer update (ID + 1) so plots and percentage milestones
    cannot shift by one. This callback never launches processes or modifies data.
    """
    queue = args.opsd_eval_queue
    step = rollout_id + 1
    stop_after = getattr(args, "debug_exit_after_rollout", None)
    final_step = args.num_rollout
    if stop_after is not None:
        final_step = min(final_step, args.start_rollout_id + stop_after)
    if step not in evaluation_steps(args.num_rollout) and step != final_step:
        return
    if queue is None:
        raise ValueError("Study save hook requires --opsd-eval-queue")
    for name in ("pending", "running", "done", "failed"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    if hf_checkpoint_dir is None:
        raise RuntimeError("Full-parameter study evaluation requires --save-hf")
    checkpoint = Path(hf_checkpoint_dir)
    digest = checkpoint_digest(checkpoint)
    run_dir = Path(args.save).parent if args.save is not None else checkpoint.parent.parent
    job_name = f"{run_dir.name}-step-{step:04d}.json"
    job = {
        "completed_updates": step,
        "rollout_id": rollout_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_kind": "full_model",
        "planned_updates": args.num_rollout,
        "temporary_snapshot": args.save is None,
    }
    if getattr(args, "wandb_opsd_profile", False) and getattr(args, "use_wandb", False):
        if not args.wandb_run_id:
            raise RuntimeError("Tracked OPSD evaluation requires the initialized trainer run ID")
        job["wandb"] = {"entity": args.wandb_team, "project": args.wandb_project, "run_id": args.wandb_run_id}
    deadline = time.monotonic() + 3600
    while True:
        _check_worker(queue)
        if time.monotonic() > deadline:
            raise TimeoutError("Evaluation queue exceeded its one-hour progress deadline")
        existing = [queue / name / job_name for name in ("pending", "running", "done")]
        if any(path.exists() for path in existing):
            for path in existing:
                if path.exists() and json.loads(path.read_text()).get("checkpoint_sha256") != digest:
                    raise ValueError("Evaluation job identity collides with a different checkpoint")
            break
        if sum(len(list((queue / name).glob("*.json"))) for name in ("pending", "running")) < 2:
            _write_json(queue / "pending" / job_name, job)
            break
        time.sleep(1)
    if step == final_step:
        while not (queue / "done" / job_name).exists():
            _check_worker(queue)
            if time.monotonic() > deadline:
                raise TimeoutError("Final checkpoint evaluation did not complete")
            time.sleep(1)


def retire_evaluation_snapshot(job: dict) -> None:
    """Delete only a successful job's owned temporary export, never its source model."""
    if not job.get("temporary_snapshot", False):
        return
    checkpoint = Path(job["checkpoint"])
    if checkpoint.is_symlink() or checkpoint.parent.name != "eval-snapshots":
        raise ValueError("Temporary evaluation snapshot is outside the study export layout")
    if checkpoint.name != f"step_{job['rollout_id']}" or not (checkpoint / ".complete").is_file():
        raise ValueError("Refusing to retire an incomplete or misidentified evaluation snapshot")
    shutil.rmtree(checkpoint)
