"""Pilot data must stay disjoint and must not truncate privileged contexts."""

from types import SimpleNamespace

import pytest

from tools.opsd.prepare_pilot import problem_id, select_examples


def _tokenizer():
    return SimpleNamespace(
        pad_token_id=-1,
        apply_chat_template=lambda messages, **kwargs: messages[0]["content"],
        encode=lambda text, **kwargs: [1] * len(text),
    )


def test_pilot_split_is_deterministic_disjoint_and_unprivileged():
    rows = [{"problem": f"Question {i}", "solution": "A short solution", "Answer": "4"} for i in range(8)]
    rows.append(rows[0] | {"problem": "  Question   0 "})
    kwargs = dict(train_count=4, eval_count=2, sequence_length=2048, response_length=128)
    train, evaluation, audit = select_examples(rows, _tokenizer(), **kwargs)
    reverse_train, reverse_eval, _ = select_examples(list(reversed(rows[:-1])), _tokenizer(), **kwargs)
    train_ids = [row["metadata"]["problem_id"] for row in train]
    eval_ids = [row["metadata"]["problem_id"] for row in evaluation]
    assert train_ids == [row["metadata"]["problem_id"] for row in reverse_train]
    assert eval_ids == [row["metadata"]["problem_id"] for row in reverse_eval]
    assert not set(train_ids) & set(eval_ids)
    assert all("solution" in row["metadata"] for row in train)
    assert all("solution" not in row["metadata"] for row in evaluation)
    assert audit["rejections"]["duplicate_problem"] == 1
    assert problem_id("café") == problem_id("cafe\u0301")


def test_pilot_fails_instead_of_truncating_or_shrinking():
    rows = [{"problem": "Q", "solution": "s" * 4096}]
    with pytest.raises(ValueError, match="Not enough"):
        select_examples(rows, _tokenizer(), train_count=1, eval_count=1, sequence_length=512, response_length=128)
