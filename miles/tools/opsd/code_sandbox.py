"""Private sandbox transport for code evaluation; no generated code runs here."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def grade_code(rows: list[dict], problems: dict) -> float:
    if queue := os.environ.get("OPSD_CODE_GRADING_QUEUE"):
        return _queued_grade(rows, problems, Path(queue))
    if python := os.environ.get("OPSD_CODE_GRADER_PYTHON"):
        result = subprocess.run(
            [python, str(Path(__file__))],
            input=json.dumps({"rows": rows, "problems": problems}),
            text=True,
            capture_output=True,
            timeout=1800,
            check=True,
        )
        payload = json.loads(result.stdout)
        _merge_results(rows, payload["results"])
        return payload["elapsed_seconds"]
    return _grade_code(rows, problems)


def _queued_grade(rows: list[dict], problems: dict, queue: Path) -> float:
    """Use a private coordinator when GPU workers cannot reach the sandbox service."""
    payload = json.dumps({"rows": rows, "problems": problems, "grader_sha256": hashlib.sha256(Path(__file__).with_name("code_sandbox_worker.py").read_bytes()).hexdigest()}, sort_keys=True)
    identity = hashlib.sha256(payload.encode()).hexdigest()
    for name in ("pending", "done", "failed"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    temporary = queue / (identity + ".tmp")
    temporary.write_text(payload)
    temporary.replace(queue / "pending" / (identity + ".json"))
    start = time.monotonic()
    while time.monotonic() - start < 1800:
        if (queue / "failed" / (identity + ".json")).exists():
            raise RuntimeError("Isolated code grading failed; evaluation is incomplete")
        path = queue / "done" / (identity + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if result["request_sha256"] != identity:
                raise ValueError("Code grading result belongs to a different response tape")
            _merge_results(rows, result["results"])
            return time.monotonic() - start
        time.sleep(1)
    raise TimeoutError("Isolated code grading did not complete within 30 minutes")


def _merge_results(rows: list[dict], graded: list[dict]) -> None:
    expected = {(row["id"], row["draw"]) for row in rows}
    if len(expected) != len(rows) or len(graded) != len(rows) or {(row["id"], row["draw"]) for row in graded} != expected:
        raise ValueError("Code grader returned missing or duplicate candidate results")
    by_draw = {(row["id"], row["draw"]): row for row in graded}
    for row in rows:
        result = by_draw[(row["id"], row["draw"])]
        for key in ("correct", "parse_failure", "grader_timeout", "test_failure"):
            if type(result[key]) is not bool:
                raise ValueError("Code grader returned a non-boolean outcome")
            row[key] = result[key]


def _grade_code(rows: list[dict], problems: dict) -> float:
    # The SDK is optional for math-only workers and lives in a separate environment.
    from e2b import Sandbox

    template = os.environ["OPSD_CODE_SANDBOX_TEMPLATE"]
    payload = [{"problem": problem, "candidates": [row for row in rows if row["id"] == task_id]} for task_id, problem in problems.items() if any(row["id"] == task_id for row in rows)]
    expected = {(row["id"], row["draw"]) for row in rows}
    if len(expected) != len(rows) or not {row["id"] for row in rows} <= set(problems):
        raise ValueError("Code evaluation needs unique draws and every problem definition")
    start = time.monotonic()
    sandbox = Sandbox.create(
        template=template,
        timeout=1800,
        allow_internet_access=False,
        metadata={"purpose": "opsd-code-evaluation"},
        request_timeout=60,
    )
    try:
        sandbox.files.write("/tmp/grader.py", Path(__file__).with_name("code_sandbox_worker.py").read_text())
        sandbox.files.write("/tmp/inputs.json", json.dumps(payload))
        result = sandbox.commands.run("python /tmp/grader.py /tmp/inputs.json /tmp/results.json", timeout=1700)
        if result.exit_code != 0:
            raise RuntimeError("Isolated code grader failed; evaluation is incomplete")
        graded = json.loads(sandbox.files.read("/tmp/results.json"))
        _merge_results(rows, graded)
    finally:
        sandbox.kill()
    return time.monotonic() - start


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    elapsed = _grade_code(payload["rows"], payload["problems"])
    keys = ("id", "draw", "correct", "parse_failure", "grader_timeout", "test_failure")
    print(json.dumps({"elapsed_seconds": elapsed, "results": [{k: row[k] for k in keys} for row in payload["rows"]]}))
