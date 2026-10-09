"""Reject incompatible evaluators and preserve the requested generation protocol."""

import asyncio
import io
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tools.opsd import evaluate_checkpoint as evaluator
from miles.utils.opsd_prompts import make_study_prefixes


class _ServerResponse:
    def __init__(self, info):
        self.info = info

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.info


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"model_path": "/different-model"}, "different local base"),
        ({"context_length": 12288}, "shorten"),
        ({"enable_lora": False}, "adapter support"),
    ],
)
def test_reject_incompatible_evaluator(override, reason):
    info = {"model_path": "/model", "context_length": 20480, "enable_lora": True, **override}
    session = type("Session", (), {"get": lambda self, url: _ServerResponse(info)})()
    with pytest.raises(ValueError, match=reason):
        asyncio.run(evaluator._server_configuration(session, "http://evaluator", "/model", 20480, True))


def test_eval_preserves_token_ids_budget_and_distinct_draw_seeds(monkeypatch):
    payloads = []

    async def post(session, url, payload):
        payloads.append(payload)
        return {"text": "answer", "meta_info": {"completion_tokens": 1, "finish_reason": {"type": "stop"}}}

    monkeypatch.setattr(evaluator, "_post", post)
    spec = json.loads((Path(__file__).resolve().parents[4] / "docs/two-node-experiment-plan.json").read_text())[
        "evaluation"
    ]
    assert spec["thinking"] is False
    assert spec["response_cap"] == 16384
    assert spec["total_context_cap"] == spec["response_cap"] + spec["templated_prefix_cap"] == 20480
    sampling = {key: spec[key] for key in ["temperature", "top_p", "top_k", "min_p"]}
    sampling["max_new_tokens"] = spec["response_cap"]

    class NonThinkingTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert enable_thinking is False and add_generation_prompt and not tokenize
            return "templated-with-thinking-disabled"

        def encode(self, text, *, add_special_tokens):
            assert text == "templated-with-thinking-disabled" and not add_special_tokens
            return [151644, 42, 151645, 198]

    prefix = make_study_prefixes(NonThinkingTokenizer(), problem="Question", context=None).student_ids

    async def sample_all():
        semaphore = asyncio.Semaphore(8)
        for draw in range(8):
            await evaluator._sample(
                None,
                "http://evaluator",
                semaphore,
                {"id": "problem", "dataset": "AMC23"},
                prefix,
                draw,
                "adapter-hash",
                io.StringIO(),
                sampling,
            )

    asyncio.run(sample_all())
    assert len({p["sampling_params"]["sampling_seed"] for p in payloads}) == 8
    for payload in payloads:
        assert payload["input_ids"] == list(prefix)
        assert payload["lora_path"] == "adapter-hash"
        assert all(payload["sampling_params"][key] == value for key, value in sampling.items())


def test_thinking_enabled_plan_fails_before_model_or_network_access(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"evaluation": {"thinking": True, "pi": False, "samples_per_question": 8}}))
    with pytest.raises(ValueError, match="non-thinking"):
        asyncio.run(evaluator._evaluate(Namespace(plan=plan, responses=8)))


@pytest.mark.parametrize("reported", ["expected", "stale"])
def test_full_checkpoint_pin_checks_every_engine_version(monkeypatch, reported):
    loaded = []

    class Client:
        def __init__(self, url):
            self.url = url

        async def update_weights_from_disk(self, *, model_path, weight_version):
            loaded.append((self.url, model_path, weight_version))
            return {"success": True}

        async def get_weight_version(self):
            return reported if self.url.endswith("b") else "expected"

    monkeypatch.setattr(evaluator, "SGLangApiClient", Client)
    call = evaluator._pin_full_checkpoint(["http://a", "http://b"], Path("checkpoint"), "expected")
    if reported == "stale":
        with pytest.raises(ValueError, match="requested full-model checkpoint"):
            asyncio.run(call)
    else:
        asyncio.run(call)
    assert loaded == [("http://a", "checkpoint", "expected"), ("http://b", "checkpoint", "expected")]


def test_mixed_checkpoint_response_is_rejected(monkeypatch):
    async def post(*args):
        return {"meta_info": {"weight_version": "stale"}}

    monkeypatch.setattr(evaluator, "_post", post)
    with pytest.raises(ValueError, match="different full-model checkpoint"):
        asyncio.run(
            evaluator._sample(
                None, "http://a", asyncio.Semaphore(1), {"id": "one"}, (1,), 0, None, io.StringIO(), {}, "expected"
            )
        )


def test_reloaded_checkpoint_retains_base_server_configuration(monkeypatch, tmp_path):
    """Exercise the caller: launch metadata stays on base after hot reloading weights."""
    plan = Path(__file__).resolve().parents[4] / "docs/two-node-experiment-plan.json"
    prompts, labels = tmp_path / "prompts.jsonl", tmp_path / "labels.jsonl"
    prompts.write_text("")
    labels.write_text("")

    class Session(_ServerResponse):
        def get(self, url):
            return _ServerResponse({"model_path": "/base", "context_length": 20480, "enable_lora": False})

    session = Session(None)
    monkeypatch.setattr(evaluator.aiohttp, "ClientSession", lambda **kwargs: session)
    monkeypatch.setattr(evaluator.aiohttp, "TCPConnector", lambda **kwargs: None)
    monkeypatch.setattr(evaluator.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "checkpoint_digest", lambda path: "new-version")
    monkeypatch.setattr(evaluator, "_provenance", lambda *args: {})
    pin, verify = AsyncMock(), AsyncMock()
    monkeypatch.setattr(evaluator, "_pin_full_checkpoint", pin)
    monkeypatch.setattr(evaluator, "_verify_full_versions", verify)
    monkeypatch.setattr(evaluator, "_generate_rows", AsyncMock(return_value=[]))
    monkeypatch.setattr(evaluator, "_grade", lambda *args: None)
    monkeypatch.setattr(evaluator, "_save_results", lambda *args: None)
    args = Namespace(
        plan=plan,
        prompts=prompts,
        labels=labels,
        checkpoint=Path("/trained"),
        model="/base",
        adapter=None,
        responses=8,
        limit_questions=None,
        output=tmp_path / "result.json",
        urls=["http://a"],
    )
    asyncio.run(evaluator._evaluate(args))
    pin.assert_awaited_once_with(args.urls, args.checkpoint, "new-version")
    verify.assert_awaited_once_with(args.urls, "new-version")
