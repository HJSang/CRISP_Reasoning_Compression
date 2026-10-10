"""Freeze full-source math splits while preserving earlier held-out groups.

The JSON config supplies private paths, pinned inputs, prior reservations, and
documented reference corrections. Benchmark labels are never an input. Run on
CPU with datasets, transformers, and datasketch==1.6.5 installed.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _rank(group, *, seed, namespace):
    return _sha256(f"math-cycle-v1:{seed}:{namespace}:{group}".encode())


def partition_rows(rows, *, reserved, prior_training, final_test, seed, validation_fraction, panel_size, screen_size):
    """Reservations override the requested fraction; no held-out group is reclaimed."""
    by_group = {row["metadata"]["problem_id"]: row for row in rows}
    if len(by_group) != len(rows):
        raise ValueError("Split input must contain one representative per duplicate group")
    if not 0 < validation_fraction < 1 or min(panel_size, screen_size) < 1:
        raise ValueError("Invalid split sizes")
    groups = set(by_group)
    validation = groups & set(reserved)
    candidates = groups - validation - set(prior_training)
    needed = max(0, math.ceil(len(groups) * validation_fraction) - len(validation))
    ordered = sorted(candidates, key=lambda group: _rank(group, seed=seed, namespace="validation"))
    if len(ordered) < needed:
        raise ValueError("Insufficient unexposed groups for the requested validation fraction")
    validation.update(ordered[:needed])
    train = groups - validation
    tune_candidates = validation - set(final_test) - set(prior_training)
    tune = sorted(tune_candidates, key=lambda group: _rank(group, seed=seed, namespace="tuning"))[:panel_size]
    if len(tune) != panel_size or len(train) < screen_size:
        raise ValueError("Insufficient groups for tuning panel or screen")
    train_order = sorted(train, key=lambda group: (group in prior_training, _rank(group, seed=seed, namespace="training")))
    if len(train - set(prior_training)) < screen_size:
        raise ValueError("Insufficient previously unused groups for the screen")
    sealed = sorted(validation - set(tune))
    return {
        "train": [by_group[group] for group in train_order],
        "validation": [by_group[group] for group in sorted(validation)],
        "validation-tuning": [by_group[group] for group in tune],
        "validation-sealed": [by_group[group] for group in sealed],
        "screen": [by_group[group] for group in train_order[:screen_size]],
    }


def _components(source, benchmarks):
    # Optional data-preparation dependencies are deferred so pure split tests need no ML runtime.
    from datasketch import MinHashLSH
    from tools.opsd.prepare_study import _canonical, _digest, _sketch

    parent, grams_by_key, exact, templates = {}, {}, {}, {}
    lsh = MinHashLSH(threshold=0.8, num_perm=128)

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    entries = [("benchmark:" + row["id"], row["problem"]) for row in benchmarks]
    entries += [(str(index), row["problem"]) for index, row in enumerate(source) if isinstance(row.get("problem"), str)]
    entries.sort(key=lambda item: (_digest(_canonical(item[1])), item[0]))
    for key, problem in entries:
        parent[key] = key
        canonical = _canonical(problem)
        template = re.sub(r"\d+(?:\.\d+)?", "#", canonical)
        sketch, grams = _sketch(problem)
        matches = [
            other for other in lsh.query(sketch)
            if len(grams & grams_by_key[other]) / len(grams | grams_by_key[other]) >= 0.8
        ]
        matches.extend(other for other in (exact.get(canonical), templates.get(template)) if other is not None)
        for other in matches:
            a, b = find(key), find(other)
            parent[max(a, b)] = min(a, b)
        # Index rejected/duplicate rows as well: they can bridge two retained groups.
        lsh.insert(key, sketch)
        grams_by_key[key] = grams
        exact[canonical], templates[template] = key, key
    members = {}
    for key, problem in entries:
        members.setdefault(find(key), []).append((key, _digest(_canonical(problem))))
    mapping = {}
    for group in members.values():
        benchmark_keys = [key for key, _ in group if key.startswith("benchmark:")]
        group_id = min(benchmark_keys) if benchmark_keys else min(digest for _, digest in group)
        mapping.update((key, group_id) for key, _ in group)
    return mapping


def _file_inputs(config):
    paths = list(config["source_arrows"]) + [config["historical_manifest"], config["benchmark_prompts"]]
    paths += [spec["path"] for spec in config["reservations"]]
    paths += config["prior_training_files"]
    paths += [spec["path"] for spec in config["corrected_reference_files"]]
    return {str(index): _sha256(Path(path).read_bytes()) for index, path in enumerate(paths)}


def _mapped_ids(files, mapping):
    result = set()
    for path in files:
        for row in _read_rows(path):
            source_row = str(row["metadata"]["source_row"])
            if source_row not in mapping:
                raise ValueError("Prior data contains an unknown source row")
            result.add(mapping[source_row])
    return result


def _corrected_references(specs, source):
    corrections = {}
    for spec in specs:
        wanted = set(spec["source_rows"])
        for row in _read_rows(spec["path"]):
            index = row["metadata"]["source_row"]
            if index not in wanted:
                continue
            if row["problem"] != source[index]["problem"]:
                raise ValueError("Reference correction changed the source problem")
            solution = row["metadata"]["solution"]
            if index in corrections and corrections[index] != solution:
                raise ValueError("Conflicting reference corrections")
            corrections[index] = solution
        if not wanted <= corrections.keys():
            raise ValueError("Reference correction is absent from its declared file")
    return corrections


def _audit_prefixes(rows, tokenizers):
    from miles.rollout.rm_hub.math_utils import extract_boxed_answer, mathd_normalize_answer
    from miles.utils.opsd_prompts import make_study_prefixes

    accepted, rejected = [], Counter()
    maxima = {name: {"student": 0, "worked": 0, "answer": 0} for name in tokenizers}
    for row in rows:
        metadata = row["metadata"]
        extracted = extract_boxed_answer(metadata["solution"])
        answer = metadata["answer"]
        gold = extract_boxed_answer(answer) if "\\boxed" in answer else answer
        if extracted is None or mathd_normalize_answer(extracted) != mathd_normalize_answer(gold):
            rejected["corrected_reference_answer_disagreement"] += 1
            continue
        lengths, failure = {}, None
        for name, tokenizer in tokenizers.items():
            worked = make_study_prefixes(tokenizer, problem=row["problem"], context=metadata["solution"])
            answer_prefix = make_study_prefixes(tokenizer, problem=row["problem"], context=answer)
            views = {"student": worked.student_ids, "worked": worked.teacher_ids, "answer": answer_prefix.teacher_ids}
            lengths[name] = {view: len(ids) for view, ids in views.items()}
            for view, ids in views.items():
                maxima[name][view] = max(maxima[name][view], len(ids))
                if len(ids) > (2048 if view == "student" else 4096):
                    failure = "context_budget"
                if tokenizer.pad_token_id in ids:
                    failure = "pad_in_prefix"
        if failure:
            rejected[failure] += 1
            continue
        metadata["prefix_lengths"] = lengths
        metadata["worked_prefix_length"] = next(iter(lengths.values()))["worked"]
        accepted.append(row)
    return accepted, dict(rejected), maxima


def _prepare(config):
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer
    from tools.opsd.prepare_study import _eligible

    source = concatenate_datasets([Dataset.from_file(path) for path in config["source_arrows"]])
    history = json.loads(Path(config["historical_manifest"]).read_text())
    if len(source) != history["source"]["rows"]:
        raise ValueError("Pinned source row count changed")
    benchmarks = _read_rows(config["benchmark_prompts"])
    if Counter(row["dataset"] for row in benchmarks) != Counter(config["benchmark_counts"]):
        raise ValueError("Unexpected benchmark suite")
    if _sha256(Path(config["benchmark_prompts"]).read_bytes()) != history["file_sha256"]["eval-prompts.jsonl"]:
        raise ValueError("Pinned benchmark prompts changed")
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for name, path in config["tokenizers"].items()
    }
    historical, rejections, old_groups = _eligible(source, benchmarks, next(iter(tokenizers.values())))
    if len(historical) != history["eligible_groups"] or rejections != history["rejections"]:
        raise ValueError("Historical eligibility no longer reproduces")
    print(json.dumps({"stage": "historical_eligibility", "groups": len(historical)}), flush=True)
    mapping = _components(source, benchmarks)
    reserved = _mapped_ids([spec["path"] for spec in config["reservations"]], mapping)
    final_test = _mapped_ids([spec["path"] for spec in config["reservations"] if spec["final_test"]], mapping)
    prior_training = _mapped_ids(config["prior_training_files"], mapping)
    invalid = {mapping[str(index)] for index in config["invalid_source_rows"]}
    invalid.update(mapping[index] for index, group in old_groups.items() if group in config["invalid_group_ids"])
    corrections = _corrected_references(config["corrected_reference_files"], source)
    selected, seen, new_rejections = [], set(), Counter()
    for row in historical:
        index = row["metadata"]["source_row"]
        group = mapping[str(index)]
        if group.startswith("benchmark:"):
            new_rejections["benchmark_component"] += 1
        elif group in invalid:
            new_rejections["documented_invalid_reference_group"] += 1
        elif group in seen:
            new_rejections["transitive_duplicate_component"] += 1
        else:
            seen.add(group)
            metadata = dict(row["metadata"], problem_id=group)
            if index in corrections:
                metadata["original_solution_sha256"] = _sha256(metadata["solution"].encode())
                metadata["solution"] = corrections[index]
                metadata["reference_corrected"] = True
            selected.append({"problem": row["problem"], "metadata": metadata})
    eligible, prefix_rejections, maxima = _audit_prefixes(selected, tokenizers)
    pools = partition_rows(eligible, reserved=reserved, prior_training=prior_training, final_test=final_test, **config["split"])
    report = {
        "schema_version": 1, "source": history["source"], "benchmark_pins": history["benchmark_pins"],
        "historical_eligible": len(historical), "historical_rejections": rejections,
        "additional_rejections": dict(new_rejections), "prefix_rejections": prefix_rejections,
        "eligible_groups": len(eligible), "reserved_groups": len(reserved),
        "eligible_reserved_groups": sum(row["metadata"]["problem_id"] in reserved for row in eligible),
        "reserved_prior_training_overlap": len(reserved & prior_training),
        "documented_invalid_groups": len(invalid), "corrected_reference_rows": len(corrections),
        "prefix_maxima": maxima, "split": config["split"], "input_sha256": _file_inputs(config),
        "counts": {name: len(rows) for name, rows in pools.items()},
        "duplicate_method": "Connected components of NFKC exact, numeric-template exact, and MinHash candidates with token-5gram Jaccard >= 0.8; all source rows indexed, including rejected rows.",
        "limits": ["Approximate lexical overlap screen cannot rule out paraphrases or pretraining exposure.",
                   "Reference final-answer agreement and documented corrections are not whole-proof review.",
                   "Sealed means withheld from this screen; earlier development exposure is retained and disclosed."],
    }
    return pools, report, mapping


def _write(output, name, rows):
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    (output / name).write_bytes(data)
    return _sha256(data)


def write_split(output, pools, report, mapping, *, config_sha256):
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen split")
    output.mkdir(parents=True)
    hashes = {name + ".jsonl": _write(output, name + ".jsonl", rows) for name, rows in pools.items()}
    for name in ("validation", "validation-tuning", "validation-sealed"):
        prompts = [{"id": row["metadata"]["problem_id"], "dataset": "Math validation", "problem": row["problem"]} for row in pools[name]]
        labels = [{"id": row["metadata"]["problem_id"], "dataset": "Math validation", "answer": row["metadata"]["answer"]} for row in pools[name]]
        hashes[name + "-prompts.jsonl"] = _write(output, name + "-prompts.jsonl", prompts)
        hashes[name + "-labels.jsonl"] = _write(output, name + "-labels.jsonl", labels)
    hashes["source-group-assignments.jsonl"] = _write(output, "source-group-assignments.jsonl", [{"source_row": key, "group": value} for key, value in sorted(mapping.items())])
    report["file_sha256"] = hashes
    report["preparation_config_sha256"] = config_sha256
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen split")
    config = json.loads(args.config.read_text())
    pools, report, mapping = _prepare(config)
    write_split(args.output, pools, report, mapping, config_sha256=_sha256(args.config.read_bytes()))


if __name__ == "__main__":
    main()
