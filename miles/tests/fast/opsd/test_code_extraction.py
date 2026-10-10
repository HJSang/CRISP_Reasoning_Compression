"""Code extraction must distinguish usage examples from competing implementations."""

import pytest

from tools.opsd.code_sandbox_worker import extract_program


def test_unique_implementation_and_usage_blocks():
    implementation = "def solve(x):\n    return x + 1\n"
    response = f"Here is the solution:\n```python\n{implementation}```\nExample:\n```python\nprint(solve(2))\n```"
    assert extract_program(response, "solve") == implementation


@pytest.mark.parametrize(
    "response",
    [
        "```python\ndef solve(x): return x\n```\n```python\ndef solve(x): return x + 1\n```",
        "def solve(x): return x\ndef solve(x): return x + 1",
        "```python\ndef solve(x): return x",
        "```python\ndef solve(x): return x\n```\n```python\nthis is invalid syntax !\n```",
        "```python\nprint(1)\n```",
    ],
)
def test_ambiguous_malformed_or_missing_implementation_is_rejected(response):
    assert extract_program(response, "solve") is None


def test_raw_program_is_unchanged():
    code = "from math import sqrt\ndef solve(x): return sqrt(x)"
    assert extract_program(code, "solve") == code
