---
title: NVIDIA GPUs
description: Get started with Miles on NVIDIA GPUs using the installation guide.
---

See the [Installation Guide](/getting-started/installation) to get started with Miles on NVIDIA GPUs.

## CRISP OPSD correctness pilot

This vendored tree adds `scripts/run_qwen3_4b_opsd.py` for the bounded frozen-base
OPSD pilot. Read the [HTML implementation and validation report](https://github.com/HJSang/CRISP_Reasoning_Compression/blob/main/docs/opsd-implementation-design.html#implementation-status)
for paired-input format, target-memory budgets, command examples and unvalidated
reproduction gates. The launcher defaults to two updates; the full study requires
a separate reviewed plan.
