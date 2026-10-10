"""Scientific pairing and admission must survive scheduler implementation changes."""

import pytest

from tools.opsd.generality_protocol import admit_cells, cell_jobs


@pytest.mark.parametrize("model", ["Qwen3-1.7B", "Qwen3-8B"])
@pytest.mark.parametrize("task", ["math", "code"])
def test_matched_forks_keep_student_and_teacher_identity(model, task):
    jobs = cell_jobs(root="/study", model_name=model, task=task, base_sha256="original")
    assert sum(job["updates"] for job in jobs) == 30
    assert 1 + sum(job["expected_evaluations"] for job in jobs) == 19
    for warmup, pi, opd in (jobs[:3], jobs[3:]):
        assert pi["initial_checkpoint"] == opd["initial_checkpoint"]
        assert pi["parent_run"] == opd["parent_run"] == warmup["run_id"]
        assert pi["dataset"] == opd["dataset"] and pi["seed"] == opd["seed"]
        assert opd["context"] == "none" and opd["ema_decay"] is None and opd["ema_source"] is None
    assert jobs[-2]["expected_ema_updates"] == 11 and jobs[-2]["ema_source"].endswith("step_6/ema-teacher")


def test_admission_requires_all_four_complete_cells_within_both_limits():
    assert admit_cells([10, 10, 20, 20], remaining_gpu_hours=96, remaining_seconds=21600)
    assert not admit_cells([20, 20, 30, 30], remaining_gpu_hours=96, remaining_seconds=21600)
    assert not admit_cells([10, 10, 50, 10], remaining_gpu_hours=96, remaining_seconds=21600)
    with pytest.raises(ValueError):
        admit_cells([10, 10, float("nan"), 20], remaining_gpu_hours=96, remaining_seconds=21600)
