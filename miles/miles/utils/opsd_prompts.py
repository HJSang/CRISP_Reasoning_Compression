"""Paired OPSD prefixes; only student response IDs cross the teacher boundary.

Prompt wording follows the non-reason-first collator in siyan-zhao/OPSD,
revision ae7d2519e94920c4eb6206c0c26de46d9c50abae.
"""

from dataclasses import dataclass


_ANSWER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_TASK_INSTRUCTIONS = {
    "math": _ANSWER_INSTRUCTION,
    "code": "Return a complete Python implementation in one Python code block, including the required function signature and imports.",
}
_TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


@dataclass(frozen=True)
class PairedPrefixes:
    student_ids: tuple[int, ...]
    teacher_ids: tuple[int, ...]


def make_opsd_prefixes(tokenizer, *, problem: str, solution: str | None) -> PairedPrefixes:
    """Apply each chat template once, then encode without added special tokens.

    These fixed thinking modes match the published recipe. No truncation is allowed:
    callers must reject oversized inputs rather than silently change the condition.
    An absent solution is permitted for unprivileged evaluation only.
    """
    student_message = f"Problem: {problem}\n\n{_ANSWER_INSTRUCTION}"
    student_ids = _encode_prefix(tokenizer, student_message, thinking=False)
    teacher_ids = ()
    if solution is not None:
        teacher_message = (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
            f"{_TRANSITION}\n{_ANSWER_INSTRUCTION}"
        )
        teacher_ids = _encode_prefix(tokenizer, teacher_message, thinking=True)
    return PairedPrefixes(student_ids=student_ids, teacher_ids=teacher_ids)


def _encode_prefix(tokenizer, content: str, *, thinking: bool) -> tuple[int, ...]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def make_study_prefixes(tokenizer, *, problem: str, context: str | None, task: str = "math") -> PairedPrefixes:
    """Matched non-thinking prefixes with a neutral wrapper for every PI arm.

    None means the exact no-PI student prefix; an empty string keeps the wrapper
    for the instruction/position diagnostic. Response IDs are appended by callers.
    """
    instruction = _TASK_INSTRUCTIONS[task]
    student = _encode_prefix(tokenizer, f"Problem: {problem}\n\n{instruction}", thinking=False)
    if context is None:
        return PairedPrefixes(student_ids=student, teacher_ids=student)
    message = (
        f"Problem: {problem}\n\nAdditional reference context:\n"
        f"=== Context Begin ===\n{context}\n=== Context End ===\n\n"
        f"Solve the problem independently using any relevant information above.\n{instruction}"
    )
    return PairedPrefixes(student_ids=student, teacher_ids=_encode_prefix(tokenizer, message, thinking=False))


def validate_response_ids(*, tokens, student_prefix, response_length: int) -> None:
    if not 0 <= response_length <= len(tokens) or len(tokens) - response_length != len(student_prefix):
        raise ValueError("OPSD generation changed the student prefix length")
    if list(tokens[: len(student_prefix)]) != list(student_prefix):
        raise ValueError("OPSD generation changed the student prefix token IDs")
