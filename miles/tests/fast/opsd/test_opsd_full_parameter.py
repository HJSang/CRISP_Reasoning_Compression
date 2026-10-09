"""A warmed full-parameter actor must coexist with an unchanged base teacher."""

import pytest
import torch

from miles.backends.megatron_utils.opsd import verify_teacher_base


def test_full_parameter_allows_checkpoint_drift_but_keeps_lora_base_check():
    snapshots = {"actor": {"weight": torch.ones(2, 3)}, "teacher": {"weight": torch.zeros(2, 3)}}
    verify_teacher_base(snapshots, full_parameter=True)
    with pytest.raises(ValueError, match="teacher base differs"):
        verify_teacher_base(snapshots)
    torch.testing.assert_close(snapshots["teacher"]["weight"], torch.zeros(2, 3))


@pytest.mark.parametrize("invalid", [torch.zeros(3, 2), torch.zeros(2, 3, dtype=torch.float64)])
def test_full_parameter_rejects_incompatible_teacher_tensor(invalid):
    snapshots = {"actor": {"weight": torch.ones(2, 3)}, "teacher": {"weight": invalid}}
    with pytest.raises(ValueError, match="incompatible tensors"):
        verify_teacher_base(snapshots, full_parameter=True)


def test_full_parameter_rejects_nonfinite_teacher():
    snapshots = {"actor": {"weight": torch.ones(1)}, "teacher": {"weight": torch.tensor([float("nan")])}}
    with pytest.raises(ValueError, match="nonfinite"):
        verify_teacher_base(snapshots, full_parameter=True)
