#!/bin/bash
# Merge FSDP checkpoints to HuggingFace format in parallel.
#
# Usage (option 1 - auto-discover all steps under base dirs):
#   CHECKPOINT_BASE_DIRS=/path/to/ckpt1/checkpoints,/path/to/ckpt2/checkpoints \
#       bash merge_checkpoints.sh
#
# Usage (option 2 - explicit list of global_step_*/actor paths):
#   CHECKPOINT_PATHS=/path/to/checkpoints/global_step_50/actor,/path/to/checkpoints/global_step_100/actor \
#       bash merge_checkpoints.sh
#
# CHECKPOINT_BASE_DIRS: comma-separated checkpoint dirs; auto-discovers all global_step_*/actor.
# CHECKPOINT_PATHS: comma-separated explicit paths to global_step_*/actor directories.
# If both are set, both are merged. At least one must be set.
# Up to 8 merges run in parallel (one per GPU on a single node).

set -euo pipefail

MAX_PARALLEL="${MAX_PARALLEL:-8}"

STEP_DIRS=()

# Option 1: Auto-discover from base dirs
if [ -n "${CHECKPOINT_BASE_DIRS:-}" ]; then
    IFS=',' read -ra BASE_DIRS <<< "$CHECKPOINT_BASE_DIRS"
    for base_dir in "${BASE_DIRS[@]}"; do
        base_dir="${base_dir## }"  # trim leading space
        base_dir="${base_dir%% }"  # trim trailing space
        for d in "${base_dir}"/global_step_*/actor; do
            [ -d "$d" ] || continue
            STEP_DIRS+=("$d")
        done
    done
fi

# Option 2: Explicit paths
if [ -n "${CHECKPOINT_PATHS:-}" ]; then
    IFS=',' read -ra EXPLICIT_PATHS <<< "$CHECKPOINT_PATHS"
    for p in "${EXPLICIT_PATHS[@]}"; do
        p="${p## }"  # trim leading space
        p="${p%% }"  # trim trailing space
        if [ -d "$p" ]; then
            STEP_DIRS+=("$p")
        else
            echo "WARNING: Path not found, skipping: $p"
        fi
    done
fi

if [ ${#STEP_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoint directories found. Set CHECKPOINT_BASE_DIRS or CHECKPOINT_PATHS."
    exit 1
fi

if [ ${#STEP_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No global_step_*/actor directories found in: ${CHECKPOINT_BASE_DIRS}"
    exit 1
fi

echo "Found ${#STEP_DIRS[@]} checkpoint(s) to merge:"
printf '  %s\n' "${STEP_DIRS[@]}"

# Merge function for a single checkpoint
merge_one() {
    local actor_dir="$1"
    local gpu_id="$2"
    local step_dir
    step_dir="$(dirname "$actor_dir")"
    local target_dir="${step_dir}/huggingface"

    if [ -d "$target_dir" ] && [ -f "$target_dir/config.json" ]; then
        echo "[GPU ${gpu_id}] SKIP ${step_dir} (already merged)"
        return 0
    fi

    echo "[GPU ${gpu_id}] Merging ${actor_dir} -> ${target_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target_dir" 2>&1 | \
        sed "s/^/[GPU ${gpu_id}] /"

    echo "[GPU ${gpu_id}] Done: ${target_dir}"
}

export -f merge_one

# Run merges in parallel, up to MAX_PARALLEL at a time
running=0
gpu_slot=0
pids=()

for actor_dir in "${STEP_DIRS[@]}"; do
    merge_one "$actor_dir" "$gpu_slot" &
    pids+=($!)
    gpu_slot=$(( (gpu_slot + 1) % MAX_PARALLEL ))
    running=$((running + 1))

    # Wait for a batch to finish before launching more
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        pids=()
        running=0
    fi
done

# Wait for remaining
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "All ${#STEP_DIRS[@]} checkpoint(s) merged successfully."
