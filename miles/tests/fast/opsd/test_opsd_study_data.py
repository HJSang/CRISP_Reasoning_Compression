"""Conservative input quarantine and deterministic paired-block balance."""

from collections import Counter

from tools.opsd.prepare_study import _contains_derivation, _stratified_blocks


def test_solved_prompt_artifacts_are_quarantined_without_using_model_correctness():
    assert _contains_derivation(r"The answer is \boxed{42}.")
    assert _contains_derivation("Find the area. " * 100 + "Thus: substitute the values. Therefore: 42.")
    assert not _contains_derivation("Find the area of a triangle with base 7 and height 12.")


def test_stratified_blocks_are_disjoint_and_preserve_sixteen_questions():
    rows = [
        {
            "problem": f"Find x in equation {i}",
            "metadata": {"problem_id": str(i), "worked_prefix_length": 100 + i * 20},
        }
        for i in range(160)
    ]
    blocks = _stratified_blocks(rows)
    assert len(blocks) == 8 and all(len(block) == 16 for block in blocks)
    keys = [row["metadata"]["problem_id"] for block in blocks for row in block]
    assert len(set(keys)) == 128
    assert keys == [row["metadata"]["problem_id"] for block in _stratified_blocks(rows) for row in block]
    counts = [Counter(row["metadata"]["difficulty_proxy"] for row in block) for block in blocks]
    for key in {key for count in counts for key in count}:
        assert max(count[key] for count in counts) - min(count[key] for count in counts) <= 1
