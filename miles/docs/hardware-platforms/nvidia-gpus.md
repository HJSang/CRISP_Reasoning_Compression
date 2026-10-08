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
a separate reviewed plan. Prepare the pinned, deduplicated 32/8 pilot split with
`tools/opsd/prepare_pilot.py`; both prefixes must fit the total context budget.
Training/sampling seeds default to 1234/42. Each update saves a checkpoint and
raw rollout token tapes under the run's output directory; keep that directory
outside the public checkout. The launcher stops its submitted job when interrupted.
With `MILES_SCRIPT_EXTERNAL_RAY=1`, cluster and node cleanup remain the external
owner's responsibility; use an isolated allocation with no GPU overlap.
