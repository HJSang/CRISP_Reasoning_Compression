"""
Prepare length-pruning OPSD data from DAPO-Math prompts.

Creates paired teacher/student prompts for reasoning length pruning via
on-policy self-distillation:
  - Student prompt: original DAPO-Math prompt used as-is (no modification)
  - Teacher prompt: raw question + conciseness instruction

The OPSD trainer then minimizes JSD(teacher || student) on student rollouts,
gradually transferring the teacher's concise reasoning style to the student.

Input: DAPO-Math-17k-dedup parquet with columns:
  - prompt: list of chat messages [{"role": "user", "content": "..."}]
  - reward_model: {"ground_truth": "...", "style": "..."}

Output: parquet with columns matching SelfDistillDataset expectations:
  - sd_prompt:  teacher prompt JSON (question + conciseness instruction)
  - sft_prompt: student prompt JSON (original DAPO-Math prompt, unchanged)
  - ground_truth: expected answer for verification
  - question: raw question text for logging
  - teacher_solution: empty (no reference solution for length pruning)

Modes:
  single: Generate one variant at a time (legacy CLI interface).
  batch:  Generate all 4 variants (concise, 20%, 50%, 80% reduction) at once
          from a single sampled subset, ensuring identical train/val splits.

Usage:
    # Batch mode (recommended) — generates all 4 variants with shared 80/20 split:
    python prepare_length_prune_data.py batch \
        --input-parquet workspace/data/DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet \
        --output-root workspace/data

    # Single variant:
    python prepare_length_prune_data.py single \
        --input-parquet workspace/data/DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet \
        --output-dir workspace/data/length_prune \
        --teacher-style concise
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# Load shared prompt config
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS = json.loads((_WORKSPACE_ROOT / "config" / "prompts.json").read_text())

# Default sampling seed (shared across all variants for reproducibility)
DEFAULT_SEED = 42


def get_original_prompt_json(prompt) -> str:
    """Return the original DAPO-Math prompt as a JSON string (unchanged)."""
    if isinstance(prompt, str):
        return prompt
    return json.dumps([dict(m) for m in prompt])


def extract_question_from_dapo(prompt) -> str:
    """Extract the raw question text from a DAPO-Math chat-format prompt."""
    if isinstance(prompt, str):
        messages = json.loads(prompt)
    else:
        messages = prompt

    content = messages[0]["content"]

    prefixes_to_strip = [
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: "
        "$Answer (without quotes) where $Answer is the answer to the problem.\n\n",
    ]
    for prefix in prefixes_to_strip:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
            break

    return content


def extract_ground_truth(reward_model) -> str:
    """Extract ground truth from the reward_model field."""
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth", "")
    return ""


def build_teacher_prompt_concise(question: str) -> str:
    """Build teacher prompt: question + conciseness instruction."""
    cfg = _PROMPTS["length_prune_teacher"]
    content = cfg["prefix"] + question + cfg["suffix"]
    return json.dumps([{"role": "user", "content": content}])


def build_teacher_prompt_percent_reduce(question: str, percent_reduce: int) -> str:
    """Build teacher prompt: question + percent reduction instruction."""
    cfg = _PROMPTS["length_prune_teacher_percent_reduce"]
    prefix = cfg["prefix"].format(percent_reduce=percent_reduce)
    content = prefix + question + cfg["suffix"]
    return json.dumps([{"role": "user", "content": content}])


def sample_and_split(
    input_parquet: str, seed: int, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DAPO data, shuffle with fixed seed, and split train/val.

    Returns (train_df, val_df) with columns: question, sft_prompt, ground_truth.
    The same seed always produces the same split.
    """
    print(f"Loading DAPO-Math data from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    print(f"  Total rows: {len(df)}")

    # Shuffle full dataset with fixed seed
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"  Shuffled with seed={seed}")

    # Extract questions and ground truths
    shuffled["question"] = shuffled["prompt"].apply(extract_question_from_dapo)
    shuffled["ground_truth"] = shuffled["reward_model"].apply(extract_ground_truth)
    shuffled["sft_prompt"] = shuffled["prompt"].apply(get_original_prompt_json)

    n_with_gt = (shuffled["ground_truth"] != "").sum()
    print(f"  {n_with_gt}/{len(shuffled)} have ground truth")

    # Split: 80% train, 20% val
    n_train = int(len(shuffled) * train_frac)
    train_df = shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled.iloc[n_train:].reset_index(drop=True)
    print(f"  Train: {len(train_df)} ({train_frac:.0%}), Val: {len(val_df)} ({1-train_frac:.0%})")

    return train_df, val_df


def apply_teacher_prompts(
    df: pd.DataFrame, teacher_style: str, percent_reduce: int = 0
) -> pd.DataFrame:
    """Add sd_prompt and teacher_solution columns for a given teacher style."""
    out = df[["question", "sft_prompt", "ground_truth"]].copy()

    if teacher_style == "concise":
        out["sd_prompt"] = out["question"].apply(build_teacher_prompt_concise)
    elif teacher_style == "percent_reduce":
        out["sd_prompt"] = out["question"].apply(
            lambda q: build_teacher_prompt_percent_reduce(q, percent_reduce)
        )
    else:
        raise ValueError(f"Unknown teacher_style: {teacher_style}")

    out["teacher_solution"] = ""
    return out[
        ["question", "sd_prompt", "sft_prompt", "teacher_solution", "ground_truth"]
    ]


def save_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    variant_name: str,
):
    """Save train/val parquets and inspection samples for one variant."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "self_distill_prompts.parquet")
    train_df.to_parquet(train_path)

    val_path = os.path.join(output_dir, "self_distill_prompts_val.parquet")
    val_df.to_parquet(val_path)

    # Save samples for inspection
    max_inspect = min(20, len(train_df))
    sample_path = os.path.join(output_dir, "length_prune_samples.json")
    samples = []
    for i in range(max_inspect):
        row = train_df.iloc[i]
        samples.append({
            "index": i,
            "question": row["question"][:300],
            "student_prompt": json.loads(row["sft_prompt"]),
            "teacher_prompt": json.loads(row["sd_prompt"]),
            "ground_truth": row["ground_truth"],
        })
    with open(sample_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"  [{variant_name}] {len(train_df)} train + {len(val_df)} val -> {output_dir}")


# -- Variant definitions --------------------------------------------------

VARIANTS = [
    {"name": "concise", "dir_suffix": "length_prune_concise", "style": "concise", "pct": 0},
    {"name": "20pct", "dir_suffix": "length_prune_20pct", "style": "percent_reduce", "pct": 20},
    {"name": "50pct", "dir_suffix": "length_prune_50pct", "style": "percent_reduce", "pct": 50},
    {"name": "80pct", "dir_suffix": "length_prune_80pct", "style": "percent_reduce", "pct": 80},
]


def cmd_batch(args):
    """Generate all 4 variants from a single shared sample."""
    train_base, val_base = sample_and_split(
        args.input_parquet, args.seed, args.train_frac
    )

    print(f"\nGenerating {len(VARIANTS)} variants...")
    for v in VARIANTS:
        train_v = apply_teacher_prompts(train_base, v["style"], v["pct"])
        val_v = apply_teacher_prompts(val_base, v["style"], v["pct"])
        out_dir = os.path.join(args.output_root, v["dir_suffix"])
        save_variant(train_v, val_v, out_dir, v["name"])

    # Preview first variant
    first_dir = os.path.join(args.output_root, VARIANTS[0]["dir_suffix"])
    preview_df = pd.read_parquet(os.path.join(first_dir, "self_distill_prompts.parquet"))
    print(f"\nPreview (concise, first 3):")
    for i in range(min(3, len(preview_df))):
        row = preview_df.iloc[i]
        teacher_msgs = json.loads(row["sd_prompt"])
        print(f"  [{i+1}] GT={row['ground_truth']}")
        print(f"       Teacher: {teacher_msgs[0]['content'][:120]}...")

    print(f"\nDone! All variants saved under {args.output_root}/")


def cmd_single(args):
    """Generate a single variant (legacy interface)."""
    train_base, val_base = sample_and_split(
        args.input_parquet, args.seed, args.train_frac
    )

    train_v = apply_teacher_prompts(train_base, args.teacher_style, args.percent_reduce)
    val_v = apply_teacher_prompts(val_base, args.teacher_style, args.percent_reduce)
    save_variant(train_v, val_v, args.output_dir, args.teacher_style)

    print(f"\nDone! {len(train_v)} train + {len(val_v)} val prompts -> {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare length-pruning OPSD data from DAPO-Math prompts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- batch sub-command ---
    p_batch = subparsers.add_parser(
        "batch", help="Generate all 4 variants (concise, 20%%, 50%%, 80%%) at once"
    )
    p_batch.add_argument(
        "--input-parquet", type=str, required=True,
        help="Path to DAPO-Math parquet",
    )
    p_batch.add_argument(
        "--output-root", type=str, required=True,
        help="Root directory (variants saved as sub-dirs)",
    )
    p_batch.add_argument("--train-frac", type=float, default=0.8, help="Fraction of data for training (default: 0.8)")
    p_batch.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_batch.set_defaults(func=cmd_batch)

    # --- single sub-command ---
    p_single = subparsers.add_parser(
        "single", help="Generate a single variant"
    )
    p_single.add_argument(
        "--input-parquet", type=str, required=True,
        help="Path to DAPO-Math parquet",
    )
    p_single.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save output parquets",
    )
    p_single.add_argument(
        "--teacher-style", type=str, default="concise",
        choices=["concise", "percent_reduce"],
    )
    p_single.add_argument("--percent-reduce", type=int, default=30)
    p_single.add_argument("--train-frac", type=float, default=0.8, help="Fraction of data for training (default: 0.8)")
    p_single.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_single.set_defaults(func=cmd_single)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
