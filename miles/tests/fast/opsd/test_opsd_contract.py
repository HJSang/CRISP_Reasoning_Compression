"""Paired-prefix, packed-position and Miles loss-boundary regression tests."""

import os
from argparse import Namespace
from dataclasses import replace

import pytest
import torch

from miles.backends.training_utils.data.opsd import teacher_batch
from miles.backends.training_utils.loss.hub.opsd import OPSDTarget, reference_loss, response_logits
from miles.backends.training_utils.loss.hub.opsd_math import OPSDLossConfig
from miles.utils.opsd_prompts import make_opsd_prefixes, validate_response_ids
from tests.opsd_reference import load_author_collator


def _batch():
    return {
        "tokens": [torch.tensor([1, 2, 3, 4]), torch.tensor([2, 1, 3, 2, 4])],
        "total_lengths": [4, 5],
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1])],
        "sample_indices": [17, 8],
        "teacher_prompt_ids": [[4, 3, 2], [1]],
    }


def _teacher_batch(data, **overrides):
    kwargs = dict(seq_length=16, local_vocab_width=5, cache_bytes=1024, microbatch_bytes=1024, micro_batch_size=1)
    return teacher_batch(data, **(kwargs | overrides))


def test_two_prefixes_one_response_and_packed_prediction_shift():
    data = _batch()
    teacher = _teacher_batch(data)
    assert [x.tolist() for x in teacher["tokens"]] == [[4, 3, 2, 3, 4], [1, 4]]
    logits = torch.arange(12).view(1, 12, 1)
    assert [x.flatten().tolist() for x in response_logits(logits, [4, 5], [2, 1])] == [[1, 2], [7]]
    assert [x.flatten().tolist() for x in response_logits(logits, [5, 2], [2, 1])] == [[2, 3], [5]]
    assert [len(x) for x in response_logits(logits, [4, 5], [0, 1])] == [0, 1]
    validate_response_ids(tokens=[1, 2, 3, 4], student_prefix=[1, 2], response_length=2)
    with pytest.raises(ValueError, match="prefix token"):
        validate_response_ids(tokens=[1, 5, 3, 4], student_prefix=[1, 2], response_length=2)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cache_bytes": 59}, "cache"),
        ({"microbatch_bytes": 39}, "microbatch"),
        ({"seq_length": 4}, "seq-length"),
        ({"micro_batch_size": 3}, "whole fixed"),
    ],
)
def test_budget_and_length_failures(overrides, message):
    with pytest.raises(ValueError, match=message):
        _teacher_batch(_batch(), **overrides)


def test_missing_prefix_duplicate_id_and_zero_mask_fail():
    for key, value, message in [
        ("teacher_prompt_ids", None, "prefix"),
        ("sample_indices", [8, 8], "unique"),
        ("loss_masks", [[0, 0], [1]], "valid tokens"),
    ]:
        with pytest.raises(ValueError, match=message):
            _teacher_batch(_batch() | {key: value})


def _loss_fixture():
    data = _batch()
    logits = torch.randn(1, 12, 7, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 5, dtype=torch.float64).log_softmax(-1)
    config = OPSDLossConfig(temperature=1, token_clip=None)
    targets = [
        OPSDTarget(17, 3, (3, 4), 1, 5, 0, teacher[:2]),
        OPSDTarget(8, 3, (4,), 1, 5, 0, teacher[2:]),
    ]
    return (
        logits,
        data | {"opsd_targets": targets, "unconcat_tokens": data["tokens"], "opsd_rollout_ids": [3, 3]},
        config,
    )


def test_reference_microbatch_mean_and_gradients_ignore_padding():
    logits, batch, config = _loss_fixture()
    loss, count = reference_loss(logits, batch, config=config, vocab_size=5)
    student = logits[0, [1, 2, 7], :5].log_softmax(-1)
    teacher = torch.cat([x.log_probs for x in batch["opsd_targets"]])
    expected = (teacher.exp() * (teacher - student)).sum() / 3
    torch.testing.assert_close(loss, expected)
    got_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(got_grad, torch.autograd.grad(expected, logits)[0])
    assert count == 3
    assert torch.count_nonzero(got_grad[0, [0, 3, 4, 5, 6, 8, 9, 10, 11]]) == 0
    assert torch.count_nonzero(got_grad[..., 5:]) == 0


def test_sample_mean_is_invariant_to_microbatch_partition():
    """Unequal response lengths must not reweight examples when tuning batches."""
    logits, batch, config = _loss_fixture()
    combined, count = reference_loss(logits, batch, config=config, vocab_size=5, reduction="sample_mean")
    separate = []
    offset = 0
    for index, total in enumerate(batch["total_lengths"]):
        one = {key: [value[index]] for key, value in batch.items()}
        loss, _ = reference_loss(logits[:, offset : offset + total], one, config=config, vocab_size=5)
        separate.append(loss)
        offset += total
    expected = torch.stack(separate).mean()
    torch.testing.assert_close(combined, expected)
    actual_grad = torch.autograd.grad(combined, logits, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, logits)[0])
    assert count == 3


def test_study_prompt_mode_and_exact_no_pi_prefix():
    from miles.utils.opsd_prompts import make_study_prefixes

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert not tokenize and add_generation_prompt and not enable_thinking
            return "<user>" + messages[0]["content"] + "<assistant><think></think>"

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return list(text.encode())

    tokenizer = Tokenizer()
    plain = make_study_prefixes(tokenizer, problem="2+2?", context=None)
    worked = make_study_prefixes(tokenizer, problem="2+2?", context="The answer is 4.")
    empty = make_study_prefixes(tokenizer, problem="2+2?", context="")
    assert plain.student_ids == plain.teacher_ids == worked.student_ids == empty.student_ids
    assert len({plain.teacher_ids, worked.teacher_ids, empty.teacher_ids}) == 3


@pytest.mark.parametrize(
    "change",
    [dict(sample_index=9), dict(rollout_id=4), dict(response_ids=(4, 3)), dict(temperature=2), dict(vocab_start=5)],
)
def test_stale_or_misaligned_targets_fail(change):
    logits, batch, config = _loss_fixture()
    batch["opsd_targets"][0] = replace(batch["opsd_targets"][0], **change)
    with pytest.raises(ValueError, match="mismatch"):
        reference_loss(logits, batch, config=config, vocab_size=5)


def test_objective_dispatch_does_not_apply_sequence_mean_scaling(monkeypatch):
    from miles.backends.training_utils import parallel
    from miles.backends.training_utils.loss.objective import loss_function

    logits, batch, config = _loss_fixture()
    monkeypatch.setattr(parallel, "_parallel_state", Namespace(tp=Namespace(rank=0, size=1, group=None)))
    args = Namespace(loss_type="opsd_loss", opsd_beta=0, opsd_temperature=1, opsd_token_clip=0, vocab_size=5)
    expected, _ = reference_loss(logits, batch, config=config, vocab_size=5)
    for microbatches in [1, 4]:
        actual, normalizer, _ = loss_function(args, batch, microbatches, logits, apply_megatron_loss_scaling=True)
        torch.testing.assert_close(actual, expected)
        assert normalizer == 1


def test_author_collator_qwen3_batch_one():
    reference_dir, tokenizer_path = os.getenv("OPSD_REFERENCE_DIR"), os.getenv("OPSD_TOKENIZER_PATH")
    if not reference_dir or not tokenizer_path:
        pytest.skip("Set OPSD_REFERENCE_DIR and OPSD_TOKENIZER_PATH for real-tokenizer parity")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    collator = load_author_collator(reference_dir)(tokenizer, max_length=4096, reason_first=False)
    for problem, solution in [("What is 2 + 2?", "Add to obtain 4."), ("求 7−3。", "7−3=4.\nTherefore \\boxed{4}.")]:
        paired = make_opsd_prefixes(tokenizer, problem=problem, solution=solution)
        original = collator([{"problem": problem, "solution": solution}])
        assert list(paired.student_ids) == original["student_prompts"][0].tolist()
        assert list(paired.teacher_ids) == original["teacher_prompts"][0].tolist()
        assert make_opsd_prefixes(tokenizer, problem=problem, solution=None).student_ids == paired.student_ids
        assert make_opsd_prefixes(tokenizer, problem=problem, solution=None).teacher_ids == ()


def test_teacher_prefix_survives_sample_serialization_and_dp_partition():
    from miles.ray.rollout.train_data_conversion import _package_shards, convert_samples_to_train_data
    from miles.utils.types import Sample

    data = _batch()
    samples = [
        Sample(index=i, tokens=tokens.tolist(), response_length=response, teacher_prompt_ids=prefix, reward=0.0)
        for i, tokens, response, prefix in zip(
            data["sample_indices"], data["tokens"], data["response_lengths"], data["teacher_prompt_ids"], strict=True
        )
    ]
    samples = [Sample.from_dict(sample.to_dict()) for sample in samples]
    args = Namespace(
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        rollout_top_logprobs_num=0,
        use_opd=False,
        multi_lora=False,
        use_dynamic_global_batch_size=False,
    )
    converted = convert_samples_to_train_data(args, samples, {}, None, None)
    shards = _package_shards(args, converted, [[1], [0]])
    assert shards[0]["sample_indices"] == [8]
    assert shards[0]["teacher_prompt_ids"] == [[1]]
    assert shards[1]["teacher_prompt_ids"] == [[4, 3, 2]]
    samples[0].teacher_prompt_ids = None
    with pytest.raises(ValueError, match="every training sample"):
        convert_samples_to_train_data(args, samples, {}, None, None)
