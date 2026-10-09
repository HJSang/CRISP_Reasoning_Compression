"""Matched non-thinking Qwen3-4B OPSD study and capacity smoke runs.

Requires a pinned local base checkpoint, filtered study JSONL, and an isolated
Ray cluster. The effective batch remains four when tuning microbatch size.
Evaluation workers consume the immutable adapter checkpoints separately.

Args:
  --model-dir / --data-dir / --output-dir: Local checkpoint, input and result roots.
  --context: none, answer, worked, unrelated, or empty-wrapper diagnostic.
  --target: frozen, current, or transported full-vocabulary distribution.
  --num-rollout: Completed optimizer updates, bounded at 64 per study job.
  --micro-batch-size: One, two, or four; sample-mean loss preserves weighting.
  --adapter-path: Optional starting native LoRA checkpoint; Adam state resets.

Example:
  MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_qwen3_4b_opsd_study.py --num-rollout 2
"""

import os
import shlex
from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    dataset_name: str = "warmup.jsonl"
    num_gpus_per_node: int = 2
    num_rollout: int = 2
    micro_batch_size: int = 2
    context: str = "worked"
    target: str = "frozen"
    adapter_path: str | None = None
    replay_path: str | None = None
    evaluation_queue: str | None = None
    seed: int = 17
    rollout_seed: int = 17
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        if self.num_nodes != 1 or self.num_gpus_per_node != 2:
            raise ValueError("Study training requires one TP2/DP1 actor per node")
        if not 1 <= self.num_rollout <= 64 or self.micro_batch_size not in (1, 2, 4):
            raise ValueError("Use 1–64 updates and microbatch 1, 2 or 4 with effective batch four")
        if self.context not in {"none", "answer", "worked", "unrelated", "empty"}:
            raise ValueError("Unknown study context")
        if self.target not in {"frozen", "current", "transported"}:
            raise ValueError("Unknown study target")


def execute(args: ScriptArgs):
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1" or not os.environ.get("RAY_ADDRESS"):
        raise ValueError("Create an isolated Ray cluster and set MILES_SCRIPT_EXTERNAL_RAY=1 and RAY_ADDRESS")
    model = shlex.quote(f"{args.model_dir}/Qwen3-4B")
    checkpoint = (
        f"--hf-checkpoint {model} --load {model} --opd-teacher-load {model} "
        "--megatron-to-hf-mode bridge --finetune --no-load-optim --no-load-rng "
        "--start-rollout-id 0 "
        f"--save {shlex.quote(f'{args.output_dir}/{args.run_id}/checkpoints')} --save-interval 1 "
    )
    if args.adapter_path is not None:
        checkpoint += f"--lora-adapter-path {shlex.quote(args.adapter_path)} "
    if args.evaluation_queue is not None:
        checkpoint += (
            "--custom-megatron-post-save-hook-path miles.utils.opsd_study.enqueue_evaluation "
            f"--opsd-eval-queue {shlex.quote(args.evaluation_queue)} "
        )
    rollout = (
        f"--prompt-data {shlex.quote(f'{args.data_dir}/{args.dataset_name}')} --input-key problem --metadata-key metadata "
        "--custom-generate-function-path miles.rollout.generate_hub.opsd.generate "
        f"--num-rollout {args.num_rollout} --rollout-batch-size 4 --n-samples-per-prompt 1 "
        "--rollout-max-response-len 4096 --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1 "
        f"--rollout-seed {args.rollout_seed} "
        f"--save-debug-rollout-data {shlex.quote(f'{args.output_dir}/{args.run_id}/rollouts/{{rollout_id}}.pt')} "
    )
    if args.replay_path is not None:
        if args.num_rollout != 1:
            raise ValueError("Capacity replay permits one update only; study updates require fresh rollouts")
        rollout += f"--load-debug-rollout-data {shlex.quote(args.replay_path)} --debug-train-only "
        rollout += "--custom-megatron-before-train-step-hook-path tools.opsd.verify_update.capture_initial_adapter "
    algorithm = (
        "--loss-type opsd_loss --disable-compute-advantages-and-returns "
        "--opsd-beta 0 --opsd-temperature 1.0 --opsd-token-clip 0 --opsd-reduction sample_mean "
        f"--opsd-context {args.context} --opsd-target {args.target} "
        f"--opsd-target-cache-gib 8 --opsd-target-microbatch-gib {2 * args.micro_batch_size} "
        "--lora-type canonical_lora --lora-rank 64 --lora-alpha 128 --lora-dropout 0 "
        "--target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj "
    )
    optimizer = (
        "--optimizer adam --lr 5e-6 --lr-decay-style constant --weight-decay 0 "
        "--adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 0.1 "
    )
    parallel = (
        "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--global-batch-size 4 --micro-batch-size {args.micro_batch_size} --seq-length 8192 "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node 2 --colocate --train-backend megatron "
    )
    inference = (
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 --sglang-lora-backend triton "
        "--sglang-enable-deterministic-inference --sglang-sampling-backend pytorch "
        "--sglang-cuda-graph-max-bs-decode 4 --sglang-max-running-requests 4 "
    )
    misc = (
        f"--seed {args.seed} --attention-dropout 0 --hidden-dropout 0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
    )
    args.create_backend().execute_train(
        train_args=checkpoint
        + rollout
        + algorithm
        + optimizer
        + parallel
        + inference
        + misc
        + U.get_default_wandb_args(__file__, run_id=args.run_id),
        megatron_model_type="qwen3-4B",
        megatron_path=args.megatron_path,
        num_gpus_per_node=2,
        job_lifetime="launcher",
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
