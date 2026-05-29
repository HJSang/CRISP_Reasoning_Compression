# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Standalone dual-path math_verify reward function.

Wired into verl through ``custom_reward_function.path`` so this repo does not
have to patch ``verl/verl/utils/reward_score/`` in place.

Dual-path scoring mirrors ``workspace/src/self_distill_hybrid/sd_verifier.py``:

    1. Primary path: regex-extract ``Answer: X`` -> wrap as ``\\boxed{X}`` ->
       HuggingFace math_verify (sympy-backed symbolic equivalence).
    2. Fallback path: math_verify on the full response, which catches in-prose
       ``\\boxed{...}`` answers that Qwen3-style post-training emits after
       ``</think>``.

Final score = 1.0 if EITHER path matches, else 0.0.

The ``compute_score`` entry point keeps verl's data_source dispatch: math /
aime / math_dapo / math_dapo_reasoning sources go through the dual-path
scorer; everything else falls through to ``verl.utils.reward_score
.default_compute_score`` so gsm8k, prime_math, code, geo3k, and search_r1
sources continue to work unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Optional

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use dual_path_math_verify, install math-verify: `pip install math-verify`.")


_MATH_DATA_SOURCES = {"math_dapo", "math", "math_dapo_reasoning"}

_VERIFY_FUNC = None


def _verify_func():
    global _VERIFY_FUNC
    if _VERIFY_FUNC is None:
        _VERIFY_FUNC = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _VERIFY_FUNC


def _verify(pred_text: str, gold: str, timeout_score: float = 0.0) -> float:
    if not pred_text or not pred_text.strip():
        return 0.0
    try:
        gold_boxed = "\\boxed{" + str(gold) + "}"
        score, _ = _verify_func()([gold_boxed], [pred_text])
        return float(score)
    except TimeoutException:
        return timeout_score
    except Exception:
        return 0.0


_ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def _extract_answer_field(response: str) -> Optional[str]:
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return None
    candidate = matches[-1].strip()
    # Strip trailing punctuation/quotes/whitespace but keep '!' so "5!" stays a
    # factorial instead of being normalized to "5".
    candidate = candidate.rstrip(".,;:?\"' \t")
    return candidate if candidate else None


def dual_path_score(model_output: str, ground_truth: str, timeout_score: float = 0.0) -> float:
    """Pure dual-path math_verify scorer. Returns 1.0 if either path matches."""
    answer_payload = _extract_answer_field(model_output)
    score_answer = 0.0
    if answer_payload is not None:
        score_answer = _verify("\\boxed{" + answer_payload + "}", ground_truth, timeout_score)

    score_boxed = _verify(model_output, ground_truth, timeout_score)

    return 1.0 if (score_answer > 0 or score_boxed > 0) else 0.0


def _is_math_source(data_source: str) -> bool:
    return data_source in _MATH_DATA_SOURCES or data_source.startswith("aime")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """verl ``custom_reward_function`` entry point.

    Dispatches math/aime sources to the dual-path math_verify scorer and
    delegates every other data_source to verl's built-in
    ``default_compute_score`` so non-math reward paths remain intact.
    """
    if _is_math_source(data_source):
        return dual_path_score(solution_str, ground_truth)

    from verl.utils.reward_score import default_compute_score

    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
