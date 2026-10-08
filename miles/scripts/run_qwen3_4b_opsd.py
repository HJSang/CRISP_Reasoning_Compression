"""Bounded Qwen3-4B OPSD correctness pilot using a frozen base teacher.

Requires a local HF Qwen3-4B checkpoint and filtered JSONL rows with raw
``problem`` and ``metadata.solution`` fields. The supplied dataset must already
fit both templated prefixes plus the response cap. No downloads or full study
are started by this recipe.

Args:
  --model-dir / --data-dir / --output-dir: Checkpoint, input and output roots.
  --num-rollout: Pilot updates (1 through 10).
  --tensor-parallel-size: Training TP (1 or 2); PP and CP stay 1.
  --num-gpus-per-node: Devices to use for colocated training and rollout.

Example:
  python scripts/run_qwen3_4b_opsd.py --model-dir /root/models --data-dir /root/datasets --num-gpus-per-node 2
"""

import shlex
from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    dataset_name: str = "opsd-pilot.jsonl"
    num_gpus_per_node: int = 2
    tensor_parallel_size: int = 1
    num_rollout: int = 2
    global_batch_size: int = 4
    response_length: int = 1024
    sequence_length: int = 4096
    target_cache_gib: float = 8.0
    target_microbatch_gib: float = 1.0
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        if not 1 <= self.num_rollout <= 10:
            raise ValueError("This correctness recipe is capped at 10 updates; review a full-study plan separately")
        if self.tensor_parallel_size not in (1, 2) or self.num_gpus_per_node % self.tensor_parallel_size:
            raise ValueError("Use TP1 or TP2 with a divisible GPU count")
        dp = self.num_nodes * self.num_gpus_per_node // self.tensor_parallel_size
        if self.global_batch_size % dp:
            raise ValueError("The global batch must be divisible by DP; reference microbatch size is one")


def execute(args: ScriptArgs):
    model = shlex.quote(f"{args.model_dir}/Qwen3-4B")
    checkpoint = (
        f"--hf-checkpoint {model} --load {model} --opd-teacher-load {model} "
        "--megatron-to-hf-mode bridge --finetune --no-load-optim --no-load-rng "
        f"--save {shlex.quote(f'{args.output_dir}/{args.run_id}/checkpoints')} --save-interval 1 "
    )
    rollout = (
        f"--prompt-data {shlex.quote(f'{args.data_dir}/{args.dataset_name}')} --input-key problem --metadata-key metadata "
        "--custom-generate-function-path miles.rollout.generate_hub.opsd.generate "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.global_batch_size} --n-samples-per-prompt 1 "
        f"--rollout-max-response-len {args.response_length} --rollout-temperature 1.1 --rollout-top-p 0.95 --rollout-top-k 20 "
    )
    algorithm = (
        "--loss-type opsd_loss --disable-compute-advantages-and-returns "
        "--opsd-beta 0 --opsd-temperature 1.1 --opsd-token-clip 0.05 "
        f"--opsd-target-cache-gib {args.target_cache_gib} --opsd-target-microbatch-gib {args.target_microbatch_gib} "
        "--lora-type canonical_lora --lora-rank 64 --lora-alpha 128 --lora-dropout 0 "
        "--target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj "
    )
    optimizer = (
        "--optimizer adam --lr 5e-6 --lr-decay-style constant --weight-decay 0 "
        "--adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 0.1 "
    )
    parallel = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--global-batch-size {args.global_batch_size} --micro-batch-size 1 --seq-length {args.sequence_length} "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --colocate --train-backend megatron "
    )
    inference = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 --sglang-lora-backend triton "
    misc = "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
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
        num_gpus_per_node=args.num_gpus_per_node,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
