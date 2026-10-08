"""Prepare a deterministic, length-filtered correctness pilot from the author dataset.

Example (outputs belong outside the source checkout):
  PYTHONPATH=. python tools/opsd/prepare_pilot.py --tokenizer /path/to/Qwen3-4B --output /path/to/pilot-data
"""

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from miles.utils.opsd_prompts import make_opsd_prefixes

DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DATASET_REVISION = "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
MODEL = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def problem_id(text):
    canonical = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(canonical.encode()).hexdigest()


def select_examples(rows, tokenizer, *, train_count, eval_count, sequence_length, response_length):
    """Hash-order unique problems, then filter both contexts without truncation."""
    groups = {}
    rejected = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row["problem"], str) or not isinstance(row["solution"], str):
            rejected["missing_text"] += 1
            continue
        key = problem_id(row["problem"])
        if key in groups:
            rejected["duplicate_problem"] += 1
            continue
        groups[key] = (index, row)
    selected = []
    for key in sorted(groups):
        index, row = groups[key]
        if not row["problem"].strip() or not row["solution"].strip():
            rejected["empty_text"] += 1
            continue
        prefixes = make_opsd_prefixes(tokenizer, problem=row["problem"], solution=row["solution"])
        if any(tokenizer.pad_token_id in ids for ids in (prefixes.student_ids, prefixes.teacher_ids)):
            rejected["pad_in_prefix"] += 1
            continue
        lengths = {"student": len(prefixes.student_ids), "teacher": len(prefixes.teacher_ids)}
        if max(lengths.values()) + response_length > sequence_length:
            rejected["context_budget"] += 1
            continue
        selected.append(
            {
                "problem": row["problem"],
                "label": row.get("Answer"),
                "metadata": {
                    "solution": row["solution"],
                    "problem_id": key,
                    "source_row": index,
                    "prefix_lengths": lengths,
                },
            }
        )
        if len(selected) == train_count + eval_count:
            break
    if len(selected) != train_count + eval_count:
        raise ValueError("Not enough unique eligible examples; do not silently shrink the pilot")
    # This holdout is a wiring diagnostic, not a clean benchmark or learning claim.
    train, evaluation = selected[:train_count], selected[train_count:]
    for row in evaluation:
        row["metadata"].pop("solution")
    return train, evaluation, {"unique_problems": len(groups), "rejections": dict(rejected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=1024)
    args = parser.parse_args()
    if (
        min(args.train_count, args.eval_count, args.response_length) <= 0
        or args.sequence_length <= args.response_length
    ):
        parser.error("Counts and lengths must be positive, with room for a prefix")
    if args.output.exists():
        parser.error("Output already exists; use a new directory to preserve the previous manifest")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=MODEL_REVISION, local_files_only=True)
    rows = load_dataset(DATASET, revision=DATASET_REVISION, split="train")
    train, evaluation, audit = select_examples(
        rows,
        tokenizer,
        train_count=args.train_count,
        eval_count=args.eval_count,
        sequence_length=args.sequence_length,
        response_length=args.response_length,
    )
    args.output.mkdir(parents=True)
    hashes = {}
    for name, examples in [("opsd-pilot.jsonl", train), ("opsd-pilot-eval.jsonl", evaluation)]:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
        (args.output / name).write_text(text)
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()
    manifest = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_rows": len(rows),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "selection": "NFKC/whitespace problem SHA256 order",
        "train_examples": len(train),
        "eval_examples": len(evaluation),
        "sequence_length": args.sequence_length,
        "response_cap": args.response_length,
        "file_sha256": hashes,
        **audit,
        "train_ids": [row["metadata"]["problem_id"] for row in train],
        "eval_ids": [row["metadata"]["problem_id"] for row in evaluation],
        "note": "Exact problem deduplication only; no near-duplicate or benchmark contamination claim.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_ids")}, indent=2))


if __name__ == "__main__":
    main()
