"""Protect the scientific pairing, evaluation independence, and bounded admission."""

import hashlib
from argparse import Namespace
from pathlib import Path

import pytest

from tools.opsd.evaluate_checkpoint import evaluation_seed
from tools.opsd.replication_protocol import SEEDS, admit_remaining, paired_jobs
from tools.opsd.serve_evaluation_queue import _pin_protocol
from miles.utils.tracking_utils.opsd_wandb import run_name


def test_legacy_seed_bank_is_unchanged_and_new_replicates_are_distinct():
    for question in ("AIME 2024:0", "AMC23:39"):
        for draw in range(8):
            expected = int.from_bytes(hashlib.sha256(f"opsd-eval-17:{question}:{draw}".encode()).digest()[:4], "big") % 2**31
            assert evaluation_seed("opsd-eval-17", question, draw) == expected
            assert len({evaluation_seed(f"opsd-replication-v1-{seed}", question, draw) for seed in SEEDS}) == 3


def test_schedule_pairs_sources_and_keeps_every_evaluation():
    jobs = [paired_jobs(seed, Path("/original"), Path("/runs")) for seed in SEEDS]
    assert sum(job.updates for group in jobs for job in group) == 45
    assert sum(job.expected_evaluations for group in jobs for job in group) + len(SEEDS) == 30
    for warm, opd, pi in jobs:
        assert warm.initial_checkpoint == "/original"
        assert opd.initial_checkpoint == pi.initial_checkpoint == f"/runs/{warm.run_id}/eval-snapshots/step_6"
        assert opd.dataset == pi.dataset and opd.seed == pi.seed == warm.seed
        assert (warm.snapshot_interval, opd.snapshot_interval, pi.snapshot_interval) == (7, 1, 1)
        assert warm.retain_final_snapshot and not opd.retain_final_snapshot and not pi.retain_final_snapshot
        assert "original-teacher OPD4" in run_name(target="frozen", context="none", seed=opd.seed, group=opd.run_id)


def test_queue_cannot_reuse_another_seed_bank_or_prompt_tape(tmp_path):
    for name in ("plan", "prompts", "labels"):
        (tmp_path / name).write_text(name)
    args = Namespace(queue=tmp_path, plan=tmp_path / "plan", prompts=tmp_path / "prompts",
                     labels=tmp_path / "labels", seed_namespace="opsd-replication-v1-29")
    first = _pin_protocol(args)
    assert _pin_protocol(args) == first
    args.seed_namespace = "opsd-replication-v1-43"
    with pytest.raises(ValueError, match="protocol changed"):
        _pin_protocol(args)
    args.seed_namespace = first["seed_namespace"]
    args.prompts.write_text("different question")
    with pytest.raises(ValueError, match="protocol changed"):
        _pin_protocol(args)


def test_admission_reserves_complete_pairs_and_time_not_only_training():
    measured = dict(used_gpu_hours=8, pair_gpu_hours=8, warmup_seconds=1200, branch_seconds=1200)
    assert admit_remaining(**measured, remaining_seconds=6000)["admitted"]
    assert not admit_remaining(**(measured | {"pair_gpu_hours": 10}), remaining_seconds=6000)["admitted"]
    assert not admit_remaining(**measured, remaining_seconds=2400)["admitted"]
    with pytest.raises(ValueError, match="finite"):
        admit_remaining(**(measured | {"pair_gpu_hours": float("nan")}), remaining_seconds=6000)
