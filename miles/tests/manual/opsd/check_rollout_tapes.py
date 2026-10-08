"""Audit trusted local Miles rollout dumps against the pinned original OPSD inputs.

Run with --rollout-dir, --tokenizer, --reference-dir and --output. Only load
rollout files produced by your own run: torch dumps may contain pickled objects.
The output contains aggregate checks, never prompt text or response token tapes.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from tests.opsd_reference import load_author_collator, load_author_input_builder


def _audit_sample(sample, tokenizer, collator, build):
    tokens = torch.tensor([sample["tokens"]])
    response_length = sample["response_length"]
    assert 0 < response_length < tokens.size(1)
    prefix_length = tokens.size(1) - response_length
    inputs = collator([{"problem": sample["prompt"], "solution": sample["metadata"]["solution"]}])
    torch.testing.assert_close(inputs["student_prompts"], tokens[:, :prefix_length], atol=0, rtol=0)
    inputs = build(
        SimpleNamespace(processing_class=tokenizer), inputs, tokens, tokens.ne(tokenizer.pad_token_id).long()
    )
    response = sample["tokens"][-response_length:]
    assert inputs["teacher_input_ids"][0].tolist() == sample["teacher_prompt_ids"] + response
    assert inputs["teacher_prompt_length"] == len(sample["teacher_prompt_ids"])
    labels = inputs["labels"][0, prefix_length:]
    assert labels.ne(-100).int().tolist() == sample["loss_mask"]
    assert inputs["teacher_attention_mask"].all()
    spans = [span for call in sample["weight_versions"] for span in call]
    assert len(spans) == 1, "The single-turn pilot must use one published adapter version"
    span = spans[0]
    assert (span["abs_start"], span["abs_end"]) == (prefix_length, tokens.size(1))
    return response_length, int(span["version"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.expected_rollouts <= 10:
        parser.error("Audit between one and ten rollout batches for this bounded pilot")
    if args.output.exists():
        parser.error("Output already exists; preserve the previous audit")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    collator = load_author_collator(args.reference_dir)(tokenizer, max_length=4096, reason_first=False)
    build = load_author_input_builder(args.reference_dir)
    results = []
    for rollout_id in range(args.expected_rollouts):
        dump = torch.load(args.rollout_dir / f"{rollout_id}.pt", map_location="cpu", weights_only=False)
        assert dump["rollout_id"] == rollout_id and len(dump["samples"]) == 4
        lengths, versions = zip(*[_audit_sample(s, tokenizer, collator, build) for s in dump["samples"]], strict=True)
        assert len(set(versions)) == 1
        if results:
            assert versions[0] == results[-1]["weight_version"] + 1
        results.append(dict(rollout_id=rollout_id, samples=4, response_lengths=lengths, weight_version=versions[0]))
    result = dict(status="passed", original_input_ids_and_masks_equal=True, rollouts=results)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
