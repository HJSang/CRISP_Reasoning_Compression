"""Real-tokenizer audit and token-in/token-out transport through the OPSD generator."""

import asyncio
import contextlib
import io
import os
from types import SimpleNamespace

import pytest

from tests.opsd_token_sanity import FEATURES, TOKENIZER_REVISION, response_tapes, run_token_sanity


@pytest.fixture
def tokenizer():
    path = os.getenv("OPSD_TOKENIZER_PATH")
    if not path:
        pytest.skip("Set OPSD_TOKENIZER_PATH for pinned real-tokenizer checks")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, revision=TOKENIZER_REVISION, local_files_only=True)


def test_original_teacher_input_ids_masks_and_boundaries(tokenizer):
    reference_dir = os.getenv("OPSD_REFERENCE_DIR")
    if not reference_dir:
        pytest.skip("Set OPSD_REFERENCE_DIR for original input-construction oracle")
    with contextlib.redirect_stdout(io.StringIO()):
        report = run_token_sanity(tokenizer, reference_dir)
    assert len(report["single"]) == 48
    native = [row for row in report["single"] if not row["pad_equals_eos"]]
    alias = [row for row in report["single"] if row["pad_equals_eos"]]
    assert all(row["masked_pad_ids_inside_teacher_prefix"] == 0 for row in native)
    assert all(row["masked_pad_ids_inside_teacher_prefix"] > 0 for row in alias)
    assert [row["author_teacher_padding_gap"] for row in report["mixed_batch"]] == [20, 14, 0, 0]


def _request(tokenizer, solution=None):
    from miles.rollout.base_types import GenerateFnInput
    from miles.utils.types import Sample

    args = SimpleNamespace(
        seq_length=4096,
        rollout_max_response_len=32,
        rollout_max_context_len=None,
        use_sampling_support_replay=False,
        rollout_temperature=1.1,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        rollout_top_logprobs_num=0,
        sglang_router_policy="round_robin",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_speculative_algorithm=None,
        lora_rank=64,
        lora_train_only=False,
    )
    sample = Sample(prompt=FEATURES[0]["problem"], metadata={"solution": solution or FEATURES[0]["solution"]})
    return GenerateFnInput(
        state=SimpleNamespace(args=args, tokenizer=tokenizer),
        sample=sample,
        sampling_params={"max_new_tokens": 32},
        evaluation=False,
    )


@pytest.mark.parametrize("tape", ["answer_eos", "noncanonical_ids", "thinking_boundary", "completion_padding"])
def test_generator_preserves_returned_ids_through_teacher_batch(tokenizer, monkeypatch, tape):
    import torch

    from miles.backends.training_utils.data.opsd import teacher_batch
    from miles.rollout.generate_hub import opsd
    from miles.utils.opsd_prompts import make_opsd_prefixes

    response = response_tapes(tokenizer)[tape]
    prefixes = make_opsd_prefixes(tokenizer, **FEATURES[0])

    async def fake_post(url, payload, *, headers):
        assert payload["input_ids"] == list(prefixes.student_ids)
        assert payload["lora_path"] == "miles_lora"
        return {
            "text": "Intentionally unrelated to token IDs.",
            "meta_info": {
                "output_token_logprobs": [[-1.0, token, None] for token in response],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(opsd, "post", fake_post)
    sample = asyncio.run(opsd.generate(_request(tokenizer))).samples
    assert sample.tokens == [*prefixes.student_ids, *response]
    assert sample.teacher_prompt_ids == list(prefixes.teacher_ids)
    data = {
        "tokens": [torch.tensor(sample.tokens)],
        "teacher_prompt_ids": [sample.teacher_prompt_ids],
        "response_lengths": [sample.response_length],
        "total_lengths": [len(sample.tokens)],
        "loss_masks": [sample.loss_mask],
        "sample_indices": [0],
    }
    teacher = teacher_batch(
        data, seq_length=4096, local_vocab_width=1, cache_bytes=2**20, microbatch_bytes=2**20, micro_batch_size=1
    )
    assert teacher["tokens"][0].tolist() == [*prefixes.teacher_ids, *response]


@pytest.mark.parametrize("case", ["eos_as_pad", "literal_pad", "overlong", "interior_pad"])
def test_generator_rejects_context_changing_inputs(tokenizer, monkeypatch, case):
    from miles.rollout.generate_hub import opsd

    calls = []
    response = [tokenizer.pad_token_id, *tokenizer.encode("4", add_special_tokens=False)]

    async def fake_post(url, payload, *, headers):
        calls.append(True)
        return {
            "text": "4",
            "meta_info": {
                "output_token_logprobs": [[-1, t, None] for t in response],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(opsd, "post", fake_post)
    solution = FEATURES[0]["solution"]
    if case == "eos_as_pad":
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif case == "literal_pad":
        solution += tokenizer.pad_token
    request = _request(tokenizer, solution)
    if case == "overlong":
        request.args.seq_length = 64
    with pytest.raises(ValueError, match="PAD|seq-length"):
        asyncio.run(opsd.generate(request))
    assert len(calls) == int(case == "interior_pad")
