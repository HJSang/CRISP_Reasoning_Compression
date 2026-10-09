"""Pin and split OPSD study data with benchmark exclusion and shared PI eligibility.

Requires datasets, transformers and datasketch==1.6.5. Writes private JSONL,
content hashes, rejection counts and duplicate-group assignments. MinHash is an
approximate lexical screen, not a guarantee against paraphrases or pretraining
exposure. Reference-answer agreement is required; reasoning still needs audit.

Example:
  PYTHONPATH=. python tools/opsd/prepare_study.py --plan plan.json --tokenizer /path/to/Qwen3-4B --output /path/to/study-data
"""

import argparse
import hashlib
import json
import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer

from miles.rollout.rm_hub.math_utils import extract_boxed_answer, mathd_normalize_answer
from miles.utils.opsd_prompts import make_study_prefixes


def _canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _contains_derivation(problem):
    """Conservative, outcome-independent quarantine of solved prompt artifacts."""
    text = _canonical(problem)
    markers = sum(marker in text for marker in ("thus:", "therefore:", "substitute", "combining", "conclusion:"))
    return "\\boxed" in text or (len(text) > 800 and markers >= 3)


def _sketch(text):
    words = re.findall(r"\w+|[^\w\s]", _canonical(text))
    grams = {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}
    sketch = MinHash(num_perm=128, seed=17)
    sketch.update_batch([gram.encode() for gram in sorted(grams)])
    return sketch, grams


def _write_rows(output, name, rows):
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (output / name).write_text(text)
    return _digest(text)


def _benchmarks(plan, output):
    prompts, labels, pins = [], [], []
    for spec in plan["benchmark_evaluation"]["datasets"]:
        rows = load_dataset(spec["id"], spec["config"], revision=spec["revision"], split=spec["split"])
        if len(rows) != spec["questions"]:
            raise ValueError(f"Unexpected question count for {spec['name']}")
        for index, row in enumerate(rows):
            problem = row.get("problem", row.get("question"))
            answer = row.get("answer", row.get("Answer"))
            if not isinstance(problem, str) or answer is None:
                raise ValueError(f"Missing problem/answer in {spec['name']}")
            key = f"{spec['name']}:{index}"
            prompts.append({"id": key, "dataset": spec["name"], "problem": problem})
            labels.append({"id": key, "answer": str(answer)})
        pins.append({"id": spec["id"], "revision": spec["revision"], "config": spec["config"], "split": spec["split"]})
    hashes = {
        "eval-prompts.jsonl": _write_rows(output, "eval-prompts.jsonl", prompts),
        "eval-labels.jsonl": _write_rows(output, "eval-labels.jsonl", labels),
    }
    return prompts, pins, hashes


def _eligible(rows, benchmarks, tokenizer):
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    texts, grams_by_id, templates, groups = {}, {}, {}, {}
    rejected = Counter()
    for row in benchmarks:
        key = "benchmark:" + row["id"]
        sketch, grams = _sketch(row["problem"])
        lsh.insert(key, sketch)
        grams_by_id[key] = grams
        texts[_canonical(row["problem"])] = key
        template = re.sub(r"\d+(?:\.\d+)?", "#", _canonical(row["problem"]))
        templates[template] = key
    ordered = sorted(enumerate(rows), key=lambda item: _digest(_canonical(str(item[1]["problem"]))))
    selected = []
    for source_index, row in ordered:
        problem, solution, answer = row.get("problem"), row.get("solution"), row.get("Answer")
        if not all(isinstance(v, str) and v.strip() for v in [problem, solution]) or answer is None:
            rejected["missing_text"] += 1
            continue
        if _contains_derivation(problem):
            rejected["solved_prompt_artifact"] += 1
            continue
        canonical = _canonical(problem)
        template = re.sub(r"\d+(?:\.\d+)?", "#", canonical)
        sketch, grams = _sketch(problem)
        matches = [
            key for key in lsh.query(sketch) if len(grams & grams_by_id[key]) / len(grams | grams_by_id[key]) >= 0.8
        ]
        matches += [key for key in [texts.get(canonical), templates.get(template)] if key is not None]
        group = min(matches) if matches else _digest(canonical)
        if matches:
            rejected[
                "benchmark_overlap" if any(key.startswith("benchmark:") for key in matches) else "duplicate_group"
            ] += 1
            groups[str(source_index)] = group
            continue
        # Index every unique problem, even when its solution will fail a later gate.
        lsh.insert(group, sketch)
        grams_by_id[group] = grams
        texts[canonical], templates[template] = group, group
        groups[str(source_index)] = group
        extracted = extract_boxed_answer(solution)
        gold = extract_boxed_answer(str(answer)) if "\\boxed" in str(answer) else str(answer)
        if extracted is None or mathd_normalize_answer(extracted) != mathd_normalize_answer(gold):
            rejected["reference_final_answer_disagreement_or_unparsed"] += 1
            continue
        prefixes = make_study_prefixes(tokenizer, problem=problem, context=solution)
        answer_prefix = make_study_prefixes(tokenizer, problem=problem, context=str(answer))
        if len(prefixes.student_ids) > 2048 or max(len(prefixes.teacher_ids), len(answer_prefix.teacher_ids)) > 4096:
            rejected["context_budget"] += 1
            continue
        if any(
            tokenizer.pad_token_id in ids
            for ids in [prefixes.student_ids, prefixes.teacher_ids, answer_prefix.teacher_ids]
        ):
            rejected["pad_in_prefix"] += 1
            continue
        selected.append(
            {
                "problem": problem,
                "metadata": {
                    "solution": solution,
                    "answer": str(answer),
                    "problem_id": group,
                    "source_row": source_index,
                    "worked_prefix_length": len(prefixes.teacher_ids),
                },
            }
        )
    return selected, dict(rejected), groups


def _add_unrelated(rows, tokenizer):
    # Donors are adaptation-only and from a different deduplicated problem group.
    failures = 0
    donors = []
    for row in rows:
        context = f"Unrelated example:\nProblem: {row['problem']}\nSolution: {row['metadata']['solution']}"
        donors.append(
            (len(tokenizer.encode(context, add_special_tokens=False)), row["metadata"]["problem_id"], context)
        )
    donors.sort()
    lengths = [item[0] for item in donors]
    for row in rows:
        length = row["metadata"]["worked_prefix_length"]
        overhead = len(make_study_prefixes(tokenizer, problem=row["problem"], context="").teacher_ids)
        center = bisect_left(lengths, length - overhead)
        candidates = sorted(
            range(max(0, center - 32), min(len(donors), center + 32)),
            key=lambda j: (abs(lengths[j] + overhead - length), donors[j][1]),
        )
        for j in candidates:
            _, donor_id, context = donors[j]
            if donor_id == row["metadata"]["problem_id"]:
                continue
            prefix = make_study_prefixes(tokenizer, problem=row["problem"], context=context).teacher_ids
            if (
                0.9 * length <= len(prefix) <= 1.1 * length
                and len(prefix) <= 4096
                and tokenizer.pad_token_id not in prefix
            ):
                row["metadata"].update(
                    unrelated_context=context, unrelated_donor_id=donor_id, unrelated_prefix_length=len(prefix)
                )
                break
        if "unrelated_context" not in row["metadata"]:
            failures += 1
    return [row for row in rows if "unrelated_context" in row["metadata"]], failures


def _stratified_blocks(rows):
    """Deterministic coarse topic and reference-length difficulty proxy strata."""
    if len(rows) < 128:
        raise ValueError("Eight paired blocks require at least 128 eligible questions")
    strata = {}
    for row in rows:
        text = row["problem"].lower()
        topic = next(
            (
                name
                for name, pattern in [
                    ("geometry", r"triangle|circle|polygon|angle|rectangle|radius|sphere"),
                    ("counting", r"probability|permutation|combination|ways|random"),
                    ("number_theory", r"divisib|prime|remainder|integer|digit"),
                ]
                if re.search(pattern, text)
            ),
            "algebra_other",
        )
        difficulty_proxy = min(3, row["metadata"]["worked_prefix_length"] // 1024)
        row["metadata"].update(topic_proxy=topic, difficulty_proxy=difficulty_proxy)
        strata.setdefault((topic, difficulty_proxy), []).append(row)
    ordered = []
    while len(ordered) < 128:
        for key in sorted(strata):
            if strata[key]:
                ordered.append(strata[key].pop(0))
    blocks = [[] for _ in range(8)]
    counts = [Counter() for _ in blocks]
    for row in ordered[:128]:
        key = (row["metadata"]["topic_proxy"], row["metadata"]["difficulty_proxy"])
        block = min((i for i in range(8) if len(blocks[i]) < 16), key=lambda i: (counts[i][key], len(blocks[i]), i))
        blocks[block].append(row)
        counts[block][key] += 1
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    benchmarks, pins, hashes = _benchmarks(plan, args.output)
    for row in benchmarks:
        if len(make_study_prefixes(tokenizer, problem=row["problem"], context=None).student_ids) > 4096:
            raise ValueError("Evaluation prefix exceeds 4096 tokens")
    source = plan["dataset"]
    rows = load_dataset(source["id"], revision=source["revision"], split="train")
    selected, rejected, groups = _eligible(rows, benchmarks, tokenizer)
    print(json.dumps({"eligible_groups": len(selected), "rejections": rejected}), flush=True)
    pools, offset = {}, 0
    for name, count in source["requested_disjoint_groups"].items():
        pools[name] = selected[offset : offset + count]
        offset += count
        if len(pools[name]) != count:
            raise ValueError(f"Only {len(selected)} eligible groups; requested {offset} by pool {name}")
    adaptation, unmatched = _add_unrelated(pools["adaptation"], tokenizer)
    if len(adaptation) < 128:
        raise ValueError("Insufficient adaptation groups with length-matched unrelated contexts")
    blocks = _stratified_blocks(adaptation)
    for name, pool in pools.items():
        hashes[name + ".jsonl"] = _write_rows(args.output, name + ".jsonl", pool)
    for block in range(8):
        name = f"block-{block:02d}.jsonl"
        hashes[name] = _write_rows(args.output, name, blocks[block])
    audit = {
        "source": {"id": source["id"], "revision": source["revision"], "rows": len(rows)},
        "benchmark_pins": pins,
        "file_sha256": hashes,
        "eligible_groups": len(selected),
        "rejections": rejected,
        "unrelated_unmatched": unmatched,
        "block_stratification": "keyword topic proxy and 1024-token reference-length bins; not calibrated difficulty",
        "duplicate_method": "NFKC exact, numeric-template exact, 128-permutation MinHash candidates plus token-5gram Jaccard>=0.8",
        "manual_reasoning_and_grader_audit": "pending",
        "source_group_assignments": groups,
    }
    (args.output / "manifest.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "source_group_assignments"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
