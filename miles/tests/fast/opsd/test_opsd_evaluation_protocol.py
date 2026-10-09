"""Reject incompatible evaluators and preserve the requested generation protocol."""

import asyncio
import io

import pytest

from tools.opsd import evaluate_checkpoint as evaluator


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
        ({"context_length": 8192}, "shorten"),
        ({"enable_lora": False}, "adapter support"),
    ],
)
def test_reject_incompatible_evaluator(override, reason):
    info = {"model_path": "/model", "context_length": 12288, "enable_lora": True, **override}
    session = type("Session", (), {"get": lambda self, url: _ServerResponse(info)})()
    with pytest.raises(ValueError, match=reason):
        asyncio.run(evaluator._server_configuration(session, "http://evaluator", "/model", 12288, True))


def test_eval_preserves_token_ids_budget_and_distinct_draw_seeds(monkeypatch):
    payloads = []

    async def post(session, url, payload):
        payloads.append(payload)
        return {"text": "answer", "meta_info": {"completion_tokens": 1, "finish_reason": {"type": "stop"}}}

    monkeypatch.setattr(evaluator, "_post", post)
    sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "max_new_tokens": 8192}
    prefix = (151644, 42, 151645, 198)

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
