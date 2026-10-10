"""EMA arithmetic, immutable anchor, precision and matched-fork contracts."""

import pytest
import torch

from miles.utils.opsd_ema import OPSDEMA


def test_fp32_accumulation_survives_scoring_rounding_and_preserves_anchor():
    original = {"w": torch.tensor([1.0], dtype=torch.bfloat16)}
    scoring = {"w": original["w"].clone()}
    ema = OPSDEMA(original, decay=0.9)
    student = {"w": torch.tensor([1.0078125], dtype=torch.bfloat16)}
    for _ in range(7):
        ema.update(student)
        ema.publish(scoring)
    expected = 0.9**7 + (1 - 0.9**7) * 1.0078125
    assert ema.tensors["w"].dtype == torch.float32
    assert ema.tensors["w"].item() == pytest.approx(expected, abs=2e-7)
    assert original["w"].item() == 1.0
    assert ema.updates == 7


def test_matched_fork_preserves_accumulator_and_counter(tmp_path):
    original = {"w": torch.zeros(3)}
    source = OPSDEMA(original, decay=0.9)
    source.update({"w": torch.ones(3)})
    path = tmp_path / "rank.pt"
    source.save(path, student_sha256="paired-student")
    continuation = OPSDEMA(original, decay=0.9)
    continuation.load(path, student_sha256="paired-student")
    continuation.update({"w": torch.full((3,), 2.0)})
    torch.testing.assert_close(continuation.tensors["w"], torch.full((3,), 0.29))
    assert continuation.updates == 2
    with pytest.raises(ValueError, match="different decay or student"):
        continuation.load(path, student_sha256="wrong-student")


def test_nonfinite_update_rejected_before_any_mutation():
    original = {"a": torch.zeros(1), "b": torch.zeros(1)}
    ema = OPSDEMA(original, decay=0.9)
    with pytest.raises(ValueError, match="finite"):
        ema.update({"a": torch.ones(1), "b": torch.tensor([float("nan")])})
    assert ema.updates == 0
    assert all(t.item() == 0 for t in ema.tensors.values())


@pytest.mark.parametrize("decay", [-1, 1, float("nan")])
def test_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        OPSDEMA({"w": torch.zeros(1)}, decay=decay)
