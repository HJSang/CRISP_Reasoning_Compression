"""HumanEval+ grading entrypoint, executed only inside an isolated CPU sandbox.

Uses the pinned EvalPlus evaluator's full base and plus inputs and its reference
timing policy. Generated programs never execute in the trainer or SDK client.
"""

import ast
import importlib.metadata
import json
import multiprocessing
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def extract_program(text: str, entry_point: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if "```" in text and text.count("```") != 2 * len(blocks):
        return None
    implementations = []
    for code in blocks or [text]:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return None
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point]
        if len(definitions) > 1:
            return None
        if definitions:
            implementations.append(code)
    # Ignore standalone usage examples, but never choose between competing answers.
    return implementations[0] if len(implementations) == 1 else None


def grade_problem(item: dict) -> list[dict]:
    # Keep parsing tests independent of the optional sandbox-only execution package.
    from evalplus.eval import untrusted_check
    from evalplus.gen.util import trusted_exec

    problem, candidates = item["problem"], item["candidates"]
    reference = problem["prompt"] + problem["canonical_solution"]
    oracle = {suite: trusted_exec(reference, problem[suite + "_input"], problem["entry_point"], record_time=True) for suite in ("base", "plus")}
    results = []
    for candidate in candidates:
        code = extract_program(candidate["text"], problem["entry_point"])
        result = {"id": candidate["id"], "draw": candidate["draw"], "parse_failure": code is None, "grader_timeout": False, "test_failure": False, "correct": False}
        if code is not None:
            statuses = []
            for suite in ("base", "plus"):
                expected, ref_time = oracle[suite]
                status, _ = untrusted_check(
                    "humaneval",
                    code,
                    problem[suite + "_input"],
                    problem["entry_point"],
                    expected=expected,
                    atol=problem["atol"],
                    ref_time=ref_time,
                    fast_check=False,
                )
                statuses.append(status)
            result.update(correct=all(status == "pass" for status in statuses), grader_timeout="timeout" in statuses, test_failure="fail" in statuses)
        results.append(result)
    return results


def main():
    if importlib.metadata.version("evalplus") != "0.0.0+26d6d00":
        raise ValueError("The grader requires the approved pinned EvalPlus sandbox image")
    payload = json.loads(Path(sys.argv[1]).read_text())
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = [row for group in pool.map(grade_problem, payload) for row in group]
    Path(sys.argv[2]).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
