# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Mirrors workspace/src/eval/scoring.py: dual-path scorer.
#   1) Primary: regex-extract "Answer: X" -> wrap as \boxed{X} -> math_verify
#   2) Fallback: math_verify on the full response (catches in-prose \boxed{...})
# Final score = 1.0 if EITHER path matches.

import re

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


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


def _extract_answer_field(response: str):
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return None
    candidate = matches[-1].strip()
    candidate = candidate.rstrip(".,;:?\"' \t")
    return candidate if candidate else None


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    # Primary path: explicit "Answer: X" extraction
    answer_payload = _extract_answer_field(model_output)
    score_answer = 0.0
    if answer_payload is not None:
        score_answer = _verify("\\boxed{" + answer_payload + "}", ground_truth, timeout_score)

    # Fallback path: math_verify scans the full response for \boxed{...} or bare expression
    score_boxed = _verify(model_output, ground_truth, timeout_score)

    return 1.0 if (score_answer > 0 or score_boxed > 0) else 0.0
