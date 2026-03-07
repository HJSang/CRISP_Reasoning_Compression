"""
Self-Distillation dataset for VERL HybridEngine.

Loads pre-prepared SD prompts (from prepare_self_distill_data.py) and returns
items compatible with VERL's DataLoader + collate_fn for AgentLoopManager
generation.

Each item contains:
  - raw_prompt: SD prompt messages (for sglang generation via AgentLoopManager)
  - sft_prompt: SFT training prompt (JSON str, used after verification)
  - ground_truth: Expected answer (for math verification)
  - question: Raw question text (for logging)
"""

import json
import logging
import os
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of sample dicts into batched tensors and arrays.

    Identical to verl.utils.dataset.rl_dataset.collate_fn — we reuse the same
    pattern so DataProto.from_single_dict works correctly downstream.
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class SelfDistillDataset(Dataset):
    """Dataset for self-distillation prompts.

    Loads the parquet produced by ``prepare_self_distill_data.py`` and returns
    items that the AgentLoopManager can consume for sglang generation.

    Required parquet columns:
        sd_prompt   – JSON string of chat-format messages for student generation
        sft_prompt  – JSON string of chat-format messages for SFT training
        ground_truth – expected answer string for verification
        question     – raw question text (for logging)

    Args:
        data_files: Path(s) to parquet file(s).
        tokenizer: HuggingFace tokenizer (used only for prompt length filtering).
        config: OmegaConf config with dataset options.
        processor: Optional multimodal processor (unused, kept for API compat).
        max_samples: Maximum number of samples to use (-1 = all).
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = list(data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/sd"))
        self.max_prompt_length = config.get("max_prompt_length", 4096)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self._download_and_read()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_and_read(self):
        from verl.utils.fs import copy_to_local

        for i, path in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=path, cache_dir=self.cache_dir)

        dfs = []
        for path in self.data_files:
            if path.endswith(".parquet"):
                df = datasets.load_dataset("parquet", data_files=path)["train"]
            elif path.endswith(".json"):
                df = datasets.load_dataset("json", data_files=path)["train"]
            else:
                raise ValueError(f"Unsupported file format: {path}")
            dfs.append(df)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dfs)

        total = len(self.dataframe)
        logger.info("Loaded %d prompts", total)

        if 0 < self.max_samples < total:
            self.dataframe = self.dataframe.select(range(self.max_samples))
            logger.info("Truncated to %d samples", self.max_samples)

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            max_len = self.max_prompt_length

            def _prompt_len(doc) -> int:
                try:
                    msgs = json.loads(doc["sd_prompt"]) if isinstance(doc["sd_prompt"], str) else doc["sd_prompt"]
                    return len(tokenizer.apply_chat_template(msgs, add_generation_prompt=True))
                except Exception:
                    return max_len + 1

            self.dataframe = self.dataframe.filter(
                lambda doc: _prompt_len(doc) <= max_len,
                num_proc=max(1, min(os.cpu_count() // 4, 8)),
                desc=f"Filtering prompts > {max_len} tokens",
            )
            logger.info("After length filter: %d", len(self.dataframe))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe[item]

        # Parse sd_prompt → list of message dicts (for AgentLoopManager)
        sd_prompt = row["sd_prompt"]
        if isinstance(sd_prompt, str):
            sd_prompt = json.loads(sd_prompt)

        return {
            # Tensor (required by DataProto – at least one tensor must exist)
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            # Non-tensor fields
            "raw_prompt": sd_prompt,  # AgentLoopManager reads this
            "sft_prompt": row["sft_prompt"],  # JSON str, for SFT batch construction
            "ground_truth": str(row.get("ground_truth", "")),  # For verification
            "question": str(row.get("question", "")),  # For logging
            "teacher_solution": str(row.get("teacher_solution", "")),  # For length metrics
        }

    # ------------------------------------------------------------------
    # Serialization (for checkpoint resume)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        if "dataframe" in state:
            del state["dataframe"]
        return state

    def resume_dataset_state(self):
        self._download_and_read()
