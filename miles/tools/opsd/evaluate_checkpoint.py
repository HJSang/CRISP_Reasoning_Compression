"""Eight-response, non-thinking evaluation against dedicated SGLang endpoints.

The caller owns the servers and pins an immutable full HF checkpoint (or a legacy
LoRA adapter) before requests begin. Prompt-only JSONL and separate labels prevent PI leakage. Results
are written atomically and include checkpoint and input content hashes. Raw
responses belong outside the public checkout.

Example:
  PYTHONPATH=. python tools/opsd/evaluate_checkpoint.py --model /path/to/Qwen3-4B --prompts eval-prompts.jsonl --labels eval-labels.jsonl --urls http://127.0.0.1:31000 --output /path/to/eval.json
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import signal
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.rollout.rm_hub import math_utils
from miles.rollout.rm_hub.math_utils import (
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    mathd_normalize_answer,
)
from miles.utils.opsd_checkpoint import checkpoint_digest
from miles.utils.opsd_prompts import make_study_prefixes


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _post(session, url, payload):
    async with session.post(url, json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"SGLang HTTP {response.status}: {body[:500]}")
        return json.loads(body)


async def _server_configuration(session, url, model, context_length, needs_adapter):
    async with session.get(url + "/server_info") as response:
        response.raise_for_status()
        info = await response.json()
    if Path(info["model_path"]).resolve() != Path(model).resolve():
        raise ValueError("Evaluator endpoint serves a different local base checkpoint")
    if info["context_length"] < context_length:
        raise ValueError("Evaluator server context would shorten the declared response budget")
    if needs_adapter and not info["enable_lora"]:
        raise ValueError("Evaluator endpoint has no adapter support")
    return {
        key: info.get(key)
        for key in [
            "context_length",
            "dtype",
            "kv_cache_dtype",
            "attention_backend",
            "sampling_backend",
            "max_running_requests",
            "enable_lora",
            "version",
        ]
    }


def evaluation_seed(namespace: str, question_id: str, draw: int) -> int:
    """Pair methods within a replicate without coupling independent replicates."""
    return int.from_bytes(hashlib.sha256(f"{namespace}:{question_id}:{draw}".encode()).digest()[:4], "big") % (2**31)


async def _sample(
    session, url, semaphore, row, prefix, draw, adapter_name, sink, sampling,
    weight_version=None, seed_namespace="opsd-eval-17",
):
    seed = evaluation_seed(seed_namespace, row["id"], draw)
    payload = {
        "input_ids": list(prefix),
        "sampling_params": {
            **sampling,
            "sampling_seed": seed,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    }
    if adapter_name is not None:
        payload["lora_path"] = adapter_name
    async with semaphore:
        start = time.monotonic()
        output = await _post(session, url + "/generate", payload)
        elapsed = time.monotonic() - start
    meta = output["meta_info"]
    if weight_version is not None and str(meta.get("weight_version")) != weight_version:
        raise ValueError("Evaluation response came from a different full-model checkpoint")
    result = {
        "id": row["id"],
        "dataset": row["dataset"],
        "draw": draw,
        "seed": seed,
        "text": output["text"],
        "tokens": meta["completion_tokens"],
        "finish_reason": meta["finish_reason"],
        "latency_seconds": elapsed,
        "weight_version": meta.get("weight_version"),
    }
    sink.write(json.dumps(result) + "\n")
    sink.flush()
    return result


async def _verify_full_versions(urls, expected):
    versions = await asyncio.gather(*[SGLangApiClient(url).get_weight_version() for url in urls])
    if not versions or any(str(version) != expected for version in versions):
        raise ValueError("Evaluator fleet did not retain the requested full-model checkpoint")


async def _pin_full_checkpoint(urls, checkpoint, version):
    results = await asyncio.gather(
        *[
            SGLangApiClient(url).update_weights_from_disk(model_path=str(checkpoint), weight_version=version)
            for url in urls
        ]
    )
    if any(not result or result.get("success") is not True for result in results):
        raise RuntimeError("Failed to load the full-model checkpoint on every evaluator")
    await _verify_full_versions(urls, version)


async def _generate_rows(session, args, prompts, prefixes, adapter_name, sink, sampling, checkpoint_hash):
    semaphores = [asyncio.Semaphore(args.concurrency) for _ in args.urls]
    requests = []
    for row, prefix in zip(prompts, prefixes, strict=True):
        for draw in range(args.responses):
            server = len(requests) % len(args.urls)
            requests.append(
                _sample(
                    session,
                    args.urls[server],
                    semaphores[server],
                    row,
                    prefix,
                    draw,
                    adapter_name,
                    sink,
                    sampling,
                    checkpoint_hash,
                    args.seed_namespace,
                )
            )
    return await asyncio.gather(*requests)


class _GradingTimeout(BaseException):
    """Escape the symbolic helper's ordinary Exception fallback on timeout."""


def _grade(rows, labels):
    previous_handler = signal.signal(signal.SIGALRM, _grade_timeout)
    try:
        for row in rows:
            signal.setitimer(signal.ITIMER_REAL, 2)
            try:
                _grade_one(row, labels[row["id"]])
            except _GradingTimeout:
                row.update(extracted_answer=None, correct=False, parse_failure=True, grader_timeout=True)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)


def _grade_timeout(*_):
    raise _GradingTimeout("Grading exceeded two seconds")


def _integer_answer(text):
    if "=" in text:
        raise ValueError("An equation inside the final box is ambiguous")
    value = Decimal(mathd_normalize_answer(text))
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError("AIME answer must be an integer")
    return int(value)


def _grade_one(row, gold):
    answer = extract_boxed_answer(row["text"])
    if "\\boxed" in gold:
        gold = extract_boxed_answer(gold)
    correct = False
    if answer is not None and gold is not None:
        if row["dataset"].startswith("AIME"):
            try:
                correct = _integer_answer(answer) == _integer_answer(gold)
            except (ValueError, InvalidOperation):
                correct = False
        else:
            correct = grade_answer_mathd(answer, gold) or grade_answer_sympy(answer, gold)
    row.update(extracted_answer=answer, correct=bool(correct), parse_failure=answer is None, grader_timeout=False)


def _provenance(args, tokenizer):
    plan = json.loads(args.plan.read_text())
    model = Path(args.model)
    return {
        "model": plan["model"],
        "datasets": plan["benchmark_evaluation"]["datasets"],
        "seed_namespace": args.seed_namespace,
        "sampling": {
            key: plan["evaluation"][key]
            for key in [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "response_cap",
                "samples_per_question",
                "thinking",
                "pi",
            ]
        },
        "template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "task": plan.get("task", "math"),
        "model_metadata_sha256": {
            name: _file_hash(model / name)
            for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
            if (model / name).exists()
        },
        "evaluator_sha256": _file_hash(Path(__file__)),
        "grader_sha256": _file_hash(Path(math_utils.__file__)),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["sglang", "transformers", "torch", "sympy", "pylatexenc"]
        },
    }


async def _evaluate(args):
    plan = json.loads(args.plan.read_text())
    spec = plan["evaluation"]
    task = plan.get("task", "math")
    if task not in {"math", "code"}:
        raise ValueError("Unknown evaluation task")
    if spec["thinking"] or spec["pi"] or spec["samples_per_question"] != args.responses:
        raise ValueError("This evaluator requires non-thinking, unprivileged, eight-response evaluation")
    sampling = {key: spec[key] for key in ["temperature", "top_p", "top_k", "min_p"]}
    sampling["max_new_tokens"] = spec["response_cap"]
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    if args.limit_questions is not None:
        prompts = prompts[: args.limit_questions]
    label_rows = list(map(json.loads, args.labels.read_text().splitlines()))
    labels = {row["id"]: row["answer"] if task == "math" else row["problem"] for row in label_rows}
    if len(labels) != len(label_rows) or len({row["id"] for row in prompts}) != len(prompts):
        raise ValueError("Evaluation inputs contain duplicate question IDs")
    if any(row["id"] not in labels for row in prompts):
        raise ValueError("Missing evaluation labels")
    checkpoint_hash = checkpoint_digest(args.checkpoint) if args.checkpoint else None
    # Every branch retains the pinned base tokenizer/template; only weights change.
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    provenance = _provenance(args, tokenizer)
    prefixes = [make_study_prefixes(tokenizer, problem=row["problem"], context=None, task=task).student_ids for row in prompts]
    if any(
        len(prefix) > spec["templated_prefix_cap"] or len(prefix) + spec["response_cap"] > spec["total_context_cap"]
        for prefix in prefixes
    ):
        raise ValueError("Evaluation prefix and response exceed the declared context budget")
    adapter_hash = _file_hash(args.adapter / "adapter_model.safetensors") if args.adapter else None
    adapter_name = "opsd-" + adapter_hash[:16] if adapter_hash else None
    timeout = aiohttp.ClientTimeout(total=1800)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sink = args.output.with_suffix(".responses.jsonl").open("x")
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        loaded = []
        try:
            if args.checkpoint:
                await _pin_full_checkpoint(args.urls, args.checkpoint, checkpoint_hash)
            provenance["engines"] = await asyncio.gather(
                *[
                    # server_info retains the launch model after a weight reload.
                    # The checkpoint digest/version checks establish dynamic identity.
                    _server_configuration(session, url, args.model, spec["total_context_cap"], bool(args.adapter))
                    for url in args.urls
                ]
            )
            if args.adapter:
                for url in args.urls:
                    result = await _post(
                        session,
                        url + "/load_lora_adapter",
                        {"lora_name": adapter_name, "lora_path": str(args.adapter)},
                    )
                    if result.get("success") is False:
                        raise RuntimeError(f"Adapter load failed: {result}")
                    loaded.append(url)
            start = time.monotonic()
            rows = await _generate_rows(
                session, args, prompts, prefixes, adapter_name, sink, sampling, checkpoint_hash
            )
            elapsed = time.monotonic() - start
            if args.checkpoint:
                await _verify_full_versions(args.urls, checkpoint_hash)
        finally:
            sink.close()
            for url in loaded:
                await _post(session, url + "/unload_lora_adapter", {"lora_name": adapter_name})
    if args.adapter and adapter_hash != _file_hash(args.adapter / "adapter_model.safetensors"):
        raise ValueError("Adapter changed during evaluation")
    if args.checkpoint and checkpoint_hash != checkpoint_digest(args.checkpoint):
        raise ValueError("Full-model checkpoint changed during evaluation")
    grading_start = time.monotonic()
    if task == "code":
        from tools.opsd.code_sandbox import grade_code

        grade_code(rows, labels)
        provenance["grader_sha256"] = _file_hash(Path(__file__).with_name("code_sandbox_worker.py"))
        provenance["code_evaluator"] = plan["code_evaluator"]
    else:
        _grade(rows, labels)
    provenance["grading_seconds"] = time.monotonic() - grading_start
    _save_results(args, rows, spec, elapsed, provenance, adapter_hash, checkpoint_hash)


def _save_results(args, rows, spec, elapsed, provenance, adapter_hash, checkpoint_hash):
    metrics = {}
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        subset = [row for row in rows if row["dataset"] == dataset]
        metrics[dataset] = {
            "avg_at_8": sum(row["correct"] for row in subset) / len(subset),
            "completions": len(subset),
            "cap_hit_fraction": sum(row["tokens"] >= spec["response_cap"] for row in subset) / len(subset),
            "parse_failure_fraction": sum(row["parse_failure"] for row in subset) / len(subset),
            "grader_timeouts": sum(row["grader_timeout"] for row in subset),
            "unexpected_thinking_delimiters": sum(
                "<think>" in row["text"] or "</think>" in row["text"] for row in subset
            ),
        }
        if "test_failure" in subset[0]:
            question_ids = {row["id"] for row in subset}
            metrics[dataset].update(
                pass_at_8=sum(any(row["correct"] for row in subset if row["id"] == q) for q in question_ids) / len(question_ids),
                test_failure_fraction=sum(row["test_failure"] for row in subset) / len(subset),
            )
    result = {
        "status": "completed",
        "adapter_sha256": adapter_hash,
        "checkpoint_sha256": checkpoint_hash,
        "prompts_sha256": _file_hash(args.prompts),
        "labels_sha256": _file_hash(args.labels),
        "elapsed_seconds": elapsed,
        "concurrency_per_server": args.concurrency,
        "servers": len(args.urls),
        "completions": len(rows),
        "responses_per_second": len(rows) / elapsed,
        "output_tokens_per_second": sum(row["tokens"] for row in rows) / elapsed,
        "mean_response_tokens": float(np.mean([row["tokens"] for row in rows])),
        "p95_latency_seconds": float(np.percentile([row["latency_seconds"] for row in rows], 95)),
        "metrics": metrics,
        "provenance": provenance,
        "responses": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "responses"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--adapter", type=Path, help="Legacy LoRA diagnostic only.")
    weights.add_argument("--checkpoint", type=Path, help="Completed full-model HF checkpoint for the study.")
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--responses", type=int, default=8)
    parser.add_argument("--seed-namespace", default="opsd-eval-17", help="Paired evaluation seed bank; never a method name.")
    parser.add_argument("--limit-questions", type=int)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.responses != 8 or args.output.exists():
        parser.error("Use positive concurrency, exactly eight responses and a fresh output path")
    asyncio.run(_evaluate(args))


if __name__ == "__main__":
    main()
