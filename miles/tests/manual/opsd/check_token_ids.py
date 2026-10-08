"""CPU-only comparison against the original OPSD collator and input assembly.

Run with --reference-dir /path/to/OPSD --tokenizer /path/to/Qwen3-4B
and --output /tmp/opsd-token-sanity.json. No model weights or GPU are required.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

from transformers import AutoTokenizer

from tests.opsd_token_sanity import TOKENIZER_REVISION, run_token_sanity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=TOKENIZER_REVISION, local_files_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_token_sanity(tokenizer, args.reference_dir)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "single_cases": len(result["single"]),
                "mixed_batch_rows": len(result["mixed_batch"]),
                "exact_single_teacher_ids": all(row["teacher_full_ids_equal"] for row in result["single"]),
            }
        )
    )


if __name__ == "__main__":
    main()
