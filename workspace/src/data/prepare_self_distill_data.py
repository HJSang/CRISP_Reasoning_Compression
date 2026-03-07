"""
Prepare self-distillation prompts from teacher SFT data.

Takes the original teacher SFT data (question + teacher solution) and creates
"reframing" prompts for the student model. Each prompt instructs the student
to read the original question and teacher's solution, then produce its own
version that is correct, consistent with the teacher's reasoning, and in its
own words.

Steps:
1. Load the teacher data parquet (with question prompts and teacher responses)
2. Filter to only correct responses (is_correct == True)
3. Build self-distillation prompts: question + teacher solution → reframing prompt
4. Preserve ground truth answers for later verification
5. Save as parquet ready for sglang generation

Usage:
    python prepare_self_distill_data.py \
        --input-parquet /path/to/teacher_responses.parquet \
        --output-dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# Load shared prompt config (workspace/config/prompts.json)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS = json.loads((_WORKSPACE_ROOT / "config" / "prompts.json").read_text())
_SD_CONFIG = _PROMPTS["self_distill"]
_SD_HINT_CONFIG = _PROMPTS["self_distill_hint"]
_SD_ANSWER_HINT_CONFIG = _PROMPTS["self_distill_answer_hint"]
_OPSD_QWEN3_STUDENT_CONFIG = _PROMPTS["opsd_qwen3_student"]
_OPSD_QWEN3_TEACHER_CONFIG = _PROMPTS["opsd_qwen3_teacher"]


def extract_question_text(prompt) -> str:
    """Extract the raw question text from a chat-format prompt.

    Args:
        prompt: Either a JSON string or list of message dicts.

    Returns:
        The content of the first user message (with instruction prefix stripped).
    """
    if isinstance(prompt, str):
        messages = json.loads(prompt)
    else:
        messages = prompt

    content = messages[0]["content"]

    # Strip common instruction prefixes to get the raw question
    prefixes_to_strip = [
        # GPT-OSS format (no <think> tags)
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: "
        "$Answer (without quotes) where $Answer is the answer to the problem.\n\n",
        # verl SFT format (with <think> tags and Answer: line)
        "Solve the following math problem step by step and the reasoning process "
        "is enclosed within <think> </think>. The last line of your response "
        "should be of the form Answer: $Answer (without quotes) where $Answer "
        "is the answer to the problem.\n\n",
        # Simple <think> prefix
        "Solve the following math problem step by step and the reasoning process "
        "is enclosed within <think> </think>.",
        # Bare prefix
        "Solve the following math problem step by step.",
    ]
    for prefix in prefixes_to_strip:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
            break

    # Also strip common suffixes
    suffixes_to_strip = [
        '\n\nRemember to put your answer on its own line after "Answer:".',
    ]
    for suffix in suffixes_to_strip:
        if content.endswith(suffix):
            content = content[: -len(suffix)].strip()
            break

    return content


def transform_teacher_response(response: str) -> str:
    """Transform GPT-OSS channel format to Qwen <think> tag format.

    GPT-OSS uses channel markers for structured output:
      <|channel|>analysis<|message|>  — starts the reasoning / analysis section
      <|end|><|start|>assistant<|channel|>final<|message|>  — ends reasoning, starts final answer

    Qwen models expect:
      <think>[reasoning]</think>[final answer with Answer: line]

    So the mapping is:
      <|channel|>analysis<|message|>  →  <think>   (open reasoning)
      <|end|><|start|>assistant<|channel|>final<|message|>  →  </think>  (close reasoning)
    """
    response = response.replace(
        "<|channel|>analysis<|message|>",
        "<think>",
    )
    response = response.replace(
        "<|end|><|start|>assistant<|channel|>final<|message|>",
        "</think>",
    )
    return response


def build_self_distill_prompt(question: str, teacher_solution: str) -> str:
    """Build a self-distillation prompt for the student model.

    Creates a chat-format prompt that presents the question and teacher's
    solution, and instructs the student to produce its own reframing.

    Args:
        question: The raw math question text.
        teacher_solution: The teacher model's full solution (with <think> tags).

    Returns:
        JSON string of chat-format messages for the student model.
    """
    user_content = (
        _SD_CONFIG["prefix"]
        + question
        + _SD_CONFIG["middle"]
        + teacher_solution
        + _SD_CONFIG["suffix"]
    )

    messages = []

    # Add system message if present
    if _SD_CONFIG.get("system"):
        messages.append({"role": "system", "content": _SD_CONFIG["system"]})

    messages.append({"role": "user", "content": user_content})

    return json.dumps(messages)


def extract_thinking_part(teacher_solution: str) -> str:
    """Extract only the thinking part from teacher's solution.

    Extracts content between <think> and </think> tags. If no tags found,
    returns empty string (will need manual inspection).

    Args:
        teacher_solution: Full teacher solution with <think> tags.

    Returns:
        The thinking content (without the tags themselves).
    """
    import re

    match = re.search(r'<think>(.*?)</think>', teacher_solution, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # If no <think> tags, return empty string as fallback
        return ""


def build_self_distill_hint_prompt(question: str, teacher_thinking: str) -> str:
    """Build a hint-based self-distillation prompt for the student model.

    Provides only the teacher's thinking as a hint, instructing the student
    to generate its own complete solution (thinking + formal answer).

    Args:
        question: The raw math question text.
        teacher_thinking: The teacher's thinking content (extracted from <think> tags).

    Returns:
        JSON string of chat-format messages for the student model.
    """
    user_content = (
        _SD_HINT_CONFIG["prefix"]
        + question
        + _SD_HINT_CONFIG["middle"]
        + teacher_thinking
        + _SD_HINT_CONFIG["suffix"]
    )

    messages = []

    # Add system message if present
    if _SD_HINT_CONFIG.get("system"):
        messages.append({"role": "system", "content": _SD_HINT_CONFIG["system"]})

    messages.append({"role": "user", "content": user_content})

    return json.dumps(messages)


def extract_answer_part(teacher_solution: str) -> str:
    """Extract only the formal answer part from teacher's solution.

    Extracts content after </think> tag (the formal solution with boxed answer).
    If no tags found, returns the full solution as fallback.

    Args:
        teacher_solution: Full teacher solution with <think> tags.

    Returns:
        The formal answer content (without the <think> section).
    """
    import re

    # Find everything after </think>
    match = re.search(r'</think>\s*(.*)', teacher_solution, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # If no </think> tag, return full solution as fallback
        return teacher_solution


def build_self_distill_answer_hint_prompt(question: str, teacher_answer: str) -> str:
    """Build an answer-hint-based self-distillation prompt for the student model.

    Provides only the teacher's formal answer (post-thinking) as a reference,
    instructing the student to generate their own thinking process and then
    rephrase the formal solution.

    Args:
        question: The raw math question text.
        teacher_answer: The teacher's formal answer content (extracted after </think>).

    Returns:
        JSON string of chat-format messages for the student model.
    """
    user_content = (
        _SD_ANSWER_HINT_CONFIG["prefix"]
        + question
        + _SD_ANSWER_HINT_CONFIG["middle"]
        + teacher_answer
        + _SD_ANSWER_HINT_CONFIG["suffix"]
    )

    messages = []

    # Add system message if present
    if _SD_ANSWER_HINT_CONFIG.get("system"):
        messages.append({"role": "system", "content": _SD_ANSWER_HINT_CONFIG["system"]})

    messages.append({"role": "user", "content": user_content})

    return json.dumps(messages)



def build_opsd_qwen3_student_prompt(question: str) -> str:
    """Build the OPSD Qwen3 student prompt (question-only, no <think> instruction).

    Qwen3 generates <think> blocks natively, so no explicit instruction is needed.
    The student prompt is a simple "solve this problem" instruction.

    Args:
        question: The raw math question text.

    Returns:
        JSON string of chat-format messages for the student model.
    """
    content = _OPSD_QWEN3_STUDENT_CONFIG["prefix"] + question + _OPSD_QWEN3_STUDENT_CONFIG["suffix"]
    messages = [{"role": "user", "content": content}]
    return json.dumps(messages)


def build_opsd_qwen3_teacher_prompt(question: str, teacher_solution: str) -> str:
    """Build the OPSD Qwen3 teacher prompt (problem + reference solution).

    Presents the problem and teacher's reference solution, then instructs the
    teacher model to solve the same problem using that reference as context.
    The suffix requests Answer: format (not \\boxed{}) so that the teacher's
    logit distribution reinforces Answer: output — aligning JSD pressure with
    the student's training objective.

    Args:
        question: The raw math question text.
        teacher_solution: The teacher model's full solution.

    Returns:
        JSON string of chat-format messages for the teacher forward pass.
    """
    user_content = (
        _OPSD_QWEN3_TEACHER_CONFIG["prefix"]
        + question
        + _OPSD_QWEN3_TEACHER_CONFIG["middle"]
        + teacher_solution
        + _OPSD_QWEN3_TEACHER_CONFIG["suffix"]
    )

    messages = []
    if _OPSD_QWEN3_TEACHER_CONFIG.get("system"):
        messages.append({"role": "system", "content": _OPSD_QWEN3_TEACHER_CONFIG["system"]})
    messages.append({"role": "user", "content": user_content})
    return json.dumps(messages)


def build_sft_prompt(question: str) -> str:
    """Build the SFT training prompt — must match existing SFT pipeline exactly.

    The existing SFT pipeline (prepare_sft_data.py) transforms the original
    teacher prompt by replacing the plain instruction prefix with the <think>
    version. We replicate that here so that the self-distillation SFT prompt
    is byte-identical to the standard SFT prompt.  The only difference in the
    final training pair is the response (student reframing vs teacher answer).

    Args:
        question: The raw math question text (instruction prefix stripped).

    Returns:
        JSON string of chat-format messages matching existing SFT format.
    """
    instruction_prefix = _PROMPTS["think_answer"]["prefix"]
    instruction_suffix = _PROMPTS["think_answer"]["suffix"]
    content = instruction_prefix + question + instruction_suffix

    messages = [{"role": "user", "content": content}]
    return json.dumps(messages)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare self-distillation prompts from teacher data"
    )
    parser.add_argument(
        "--input-parquet",
        type=str,
        required=True,
        help="Path to input parquet with teacher-generated responses",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save self-distillation prompts",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max number of samples to use (0 = all). Useful for debug runs.",
    )
    parser.add_argument(
        "--hint-mode",
        action="store_true",
        help="Use hint-based self-distillation (provide only teacher's thinking as hint, not full solution)",
    )
    parser.add_argument(
        "--answer-hint-mode",
        action="store_true",
        help="Use answer-hint-based self-distillation (provide only teacher's formal answer as hint, not thinking)",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="default",
        choices=["default", "opsd_qwen3"],
        help=(
            "Prompt style for sd_prompt and sft_prompt construction. "
            "'default' uses self_distill/think_answer prompts (Qwen2.5 style). "
            "'opsd_qwen3' uses opsd_qwen3_teacher/opsd_qwen3_student prompts "
            "(no <think> instruction; no \\\\boxed{} in teacher to prevent format collapse)."
        ),
    )
    args = parser.parse_args()

    # Validate mutually exclusive modes
    if args.hint_mode and args.answer_hint_mode:
        parser.error("--hint-mode and --answer-hint-mode are mutually exclusive")
    if args.prompt_style == "opsd_qwen3" and (args.hint_mode or args.answer_hint_mode):
        parser.error("--prompt-style opsd_qwen3 is incompatible with --hint-mode / --answer-hint-mode")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading teacher data from {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    print(f"  Total rows: {len(df)}")

    # Filter correct responses
    if "is_correct" in df.columns:
        df_correct = df[df["is_correct"] == True].copy()
        print(f"  Correct responses: {len(df_correct)}")
    else:
        print("  WARNING: No 'is_correct' column found, using all rows")
        df_correct = df.copy()

    if len(df_correct) == 0:
        print("ERROR: No correct responses found. Exiting.")
        return

    # Optionally limit samples (for debug / test runs)
    if args.max_samples > 0 and len(df_correct) > args.max_samples:
        df_correct = df_correct.head(args.max_samples).copy()
        print(f"  Truncated to {len(df_correct)} samples (--max-samples={args.max_samples})")

    # Extract question text from prompts
    print("Extracting questions...")
    df_correct["question"] = df_correct["prompt"].apply(extract_question_text)

    # Transform teacher responses (normalize format)
    print("Transforming teacher responses...")
    df_correct["teacher_solution"] = df_correct["response"].apply(
        transform_teacher_response
    )

    # Build self-distillation prompts (for student generation)
    if args.prompt_style == "opsd_qwen3":
        print("Building OPSD Qwen3 prompts (teacher: problem+solution, student: question-only)...")
        df_correct["sd_prompt"] = df_correct.apply(
            lambda row: build_opsd_qwen3_teacher_prompt(row["question"], row["teacher_solution"]),
            axis=1,
        )
        df_correct["sft_prompt"] = df_correct["question"].apply(build_opsd_qwen3_student_prompt)
    elif args.hint_mode:
        print("Building hint-based self-distillation prompts (thinking only)...")
        # Extract only thinking part from teacher solutions
        df_correct["teacher_thinking"] = df_correct["teacher_solution"].apply(
            extract_thinking_part
        )
        # Build hint-based prompts
        df_correct["sd_prompt"] = df_correct.apply(
            lambda row: build_self_distill_hint_prompt(row["question"], row["teacher_thinking"]),
            axis=1,
        )
        df_correct["sft_prompt"] = df_correct["question"].apply(build_sft_prompt)
    elif args.answer_hint_mode:
        print("Building answer-hint-based self-distillation prompts (formal answer only)...")
        # Extract only formal answer part from teacher solutions
        df_correct["teacher_answer"] = df_correct["teacher_solution"].apply(
            extract_answer_part
        )
        # Build answer-hint-based prompts
        df_correct["sd_prompt"] = df_correct.apply(
            lambda row: build_self_distill_answer_hint_prompt(row["question"], row["teacher_answer"]),
            axis=1,
        )
        df_correct["sft_prompt"] = df_correct["question"].apply(build_sft_prompt)
    else:
        print("Building full self-distillation prompts (complete solution)...")
        df_correct["sd_prompt"] = df_correct.apply(
            lambda row: build_self_distill_prompt(row["question"], row["teacher_solution"]),
            axis=1,
        )
        # Build SFT training prompts (must match existing SFT pipeline format)
        print("Building SFT training prompts (matching existing SFT format)...")
        df_correct["sft_prompt"] = df_correct["question"].apply(build_sft_prompt)

    # Extract ground truth for verification
    ground_truths = []
    for _, row in df_correct.iterrows():
        reward = row.get("reward_model", {})
        if isinstance(reward, dict):
            ground_truths.append(reward.get("ground_truth", ""))
        else:
            ground_truths.append("")
    df_correct["ground_truth"] = ground_truths

    # Select output columns
    output_df = df_correct[
        ["question", "sd_prompt", "sft_prompt", "teacher_solution", "ground_truth"]
    ].reset_index(drop=True)

    # Random 90/10 train/val split (same ratio as prepare_sft_data.py)
    print("Splitting data 90% train / 10% val...")
    output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(output_df) * 0.9)
    train_df = output_df.iloc[:split_idx].reset_index(drop=True)
    val_df = output_df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # Save train split (used for SD generation + SFT training)
    train_path = os.path.join(args.output_dir, "self_distill_prompts.parquet")
    train_df.to_parquet(train_path)
    print(f"  Saved train -> {train_path}")

    # Save val split (used for val loss computation with teacher solutions)
    val_path = os.path.join(args.output_dir, "self_distill_prompts_val.parquet")
    val_df.to_parquet(val_path)
    print(f"  Saved val -> {val_path}")

    # Save sampled examples for inspection (complete prompts, not all)
    max_inspect = min(50, len(train_df))
    sample_path = os.path.join(args.output_dir, "self_distill_samples.json")
    samples = []
    for i in range(max_inspect):
        row = train_df.iloc[i]
        samples.append({
            "index": i,
            "question": row["question"],
            "sd_prompt": json.loads(row["sd_prompt"]),
            "sft_prompt": json.loads(row["sft_prompt"]),
            "teacher_solution": row["teacher_solution"],
            "ground_truth": row["ground_truth"],
        })
    with open(sample_path, "w") as f:
        json.dump(samples, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved {len(samples)}/{len(output_df)} sample prompts -> {sample_path}")

    # Log a few sample prompts to stdout for quick sanity check
    num_preview = min(3, len(train_df))
    print(f"\n  Preview ({num_preview}/{len(train_df)} train prompts):")
    for i in range(num_preview):
        row = train_df.iloc[i]
        print(f"    [{i+1}] GT={row['ground_truth']}  question={row['question'][:120]}...")

    print(f"\n  Inspection samples saved to {sample_path}")
    print(f"  Done! {len(train_df)} train + {len(val_df)} val prompts ready for self-distillation.")


if __name__ == "__main__":
    main()
