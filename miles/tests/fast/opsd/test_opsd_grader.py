"""Hand-labelled format and ambiguity cases for the benchmark grader."""

import pytest

from tools.opsd.evaluate_checkpoint import _grade


@pytest.mark.parametrize(
    "dataset,text,gold,expected",
    [
        ("AIME 2024", r"Answer: \boxed{045}", "45", True),
        ("AIME 2024", r"Answer: \boxed{45.0}", "045", True),
        ("AIME 2025", r"Answer: \boxed{\text{45}}", "45", True),
        ("AIME 2025", r"Answer: \boxed{45.5}", "45", False),
        ("AIME 2025", r"Answer: \boxed{p = 45}", "45", False),
        ("AIME 2025", r"Answer: \boxed{44} then \boxed{45}", "45", True),
        ("AMC23", r"Answer: \boxed{45} then \boxed{44}", "45", False),
        ("AMC23", r"Answer: \boxed{45^\circ}", "45", True),
        ("AMC23", r"Answer: \boxed{45", "45", False),
        ("AMC23", r"Answer: 45", "45", False),
    ],
)
def test_hand_labelled_grader_cases(dataset, text, gold, expected):
    rows = [{"id": "one", "dataset": dataset, "text": text}]
    _grade(rows, {"one": gold})
    assert rows[0]["correct"] is expected
    assert not rows[0]["grader_timeout"]
