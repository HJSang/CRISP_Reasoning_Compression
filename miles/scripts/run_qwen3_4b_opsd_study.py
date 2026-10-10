"""Full-parameter, non-thinking dense Qwen3 OPSD study and capacity smoke runs.

Requires a pinned local base checkpoint, filtered study JSONL, and an isolated
Ray cluster. The effective batch remains four when tuning microbatch size.
Evaluation workers consume immutable full HF checkpoints separately. Previous
LoRA capacity measurements do not validate this full-parameter recipe.
Set WANDB_ENTITY and WANDB_PROJECT after authenticating the SDK on every worker
to enable the restricted OPSD tracking profile; credentials never enter argv.

Args:
  --model-dir / --data-dir / --output-dir: Local checkpoint, input and result roots.
  --context: none, answer, worked, unrelated, or empty-wrapper diagnostic.
  --target: frozen, current, or transported full-vocabulary distribution.
  --num-rollout: Completed updates; at most 64 normally or 88 for the cyclic screen.
  --micro-batch-size: One, two, or four; sample-mean loss preserves weighting.
  --initial-checkpoint: Optional starting full HF or native checkpoint; Adam resets.
  --stop-after-rollout: Stop this job early while retaining the planned eval cadence.
  --save-checkpoints: Opt in to recovery saves; otherwise only temporary eval snapshots.
  --snapshot-interval: Export interval; seven for a seven-update warm-up, one for branches.
  --retain-final-snapshot: Keep the final HF snapshot until paired branches have consumed it.
  --model-name: Qwen3-1.7B, Qwen3-4B, or Qwen3-8B; the legacy entrypoint name stays stable.
  --task: Math or code prompt contract.
  --teacher-ema-decay / --ema-source: Optional EMA policy and its matched FP32 warm-up state.
  --cyclic-teacher-policy: Fixed original or refresh after each completed PI/OPD cycle.
  --cyclic-pi-updates / --cyclic-opd-updates: Positive phase lengths.
  --cyclic-optimizer-policy: Reset Adam each phase (default), or carry moments and counters.
  --cyclic-lr-schedule: Constant (default), or global linear 5e-6 to 5e-7 on applied updates.

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
    model_name: str = "Qwen3-4B"
    task: str = "math"
    teacher_ema_decay: float | None = None
    ema_source: str | None = None
    cyclic_teacher_policy: str | None = None
    cyclic_pi_updates: int = 7
    cyclic_opd_updates: int = 4
    cyclic_optimizer_policy: str = "reset_each_phase"
    cyclic_lr_schedule: str = "constant"
    data_dir: str = "/root/datasets"
    dataset_name: str = "warmup.jsonl"
    num_gpus_per_node: int = 2
    num_rollout: int = 2
    stop_after_rollout: int | None = None
    micro_batch_size: int = 1
    context: str = "worked"
    target: str = "frozen"
    initial_checkpoint: str | None = None
    replay_path: str | None = None
    capture_full_update: bool = False
    evaluation_queue: str | None = None
    save_checkpoints: bool = False
    snapshot_interval: int = 1
    retain_final_snapshot: bool = False
    seed: int = 17
    rollout_seed: int = 17
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        if self.model_name not in {"Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"} or self.task not in {"math", "code"}:
            raise ValueError("Use a supported Qwen3 model and math/code task")
        if self.num_nodes != 1 or self.num_gpus_per_node != 2:
            raise ValueError("Study training requires one TP2/DP1 actor per node")
        limit = 88 if self.cyclic_teacher_policy is not None else 64
        if not 1 <= self.num_rollout <= limit or self.micro_batch_size not in (1, 2, 4):
            raise ValueError(f"Use 1–{limit} updates and microbatch 1, 2 or 4 with effective batch four")
        if self.stop_after_rollout is not None and not 1 <= self.stop_after_rollout <= self.num_rollout:
            raise ValueError("Early stopping must be within the planned optimizer updates")
        if self.context not in {"none", "answer", "worked", "unrelated", "empty"}:
            raise ValueError("Unknown study context")
        if self.target not in {"frozen", "current", "transported"}:
            raise ValueError("Unknown study target")
        if not 1 <= self.snapshot_interval <= self.num_rollout:
            raise ValueError("Snapshot interval must fit the planned updates")
        if self.retain_final_snapshot and self.evaluation_queue is None:
            raise ValueError("Retaining a paired source requires its evaluation queue")
        if self.cyclic_teacher_policy is not None:
            self._validate_cyclic()
        elif self.cyclic_optimizer_policy != "reset_each_phase" or self.cyclic_lr_schedule != "constant":
            raise ValueError("Cyclic optimizer/LR policies require a cyclic teacher policy")

    def _validate_cyclic(self):
        if self.cyclic_teacher_policy not in {"fixed_original", "cycle_refresh"}:
            raise ValueError("Unknown cyclic teacher policy")
        if self.cyclic_optimizer_policy not in {"reset_each_phase", "carry"}:
            raise ValueError("Unknown cyclic optimizer policy")
        if self.cyclic_lr_schedule not in {"constant", "global_linear"}:
            raise ValueError("Unknown cyclic LR schedule")
        if min(self.cyclic_pi_updates, self.cyclic_opd_updates) < 1:
            raise ValueError("Cyclic phase lengths must be positive")
        if self.num_rollout % (self.cyclic_pi_updates + self.cyclic_opd_updates):
            raise ValueError("A cyclic job must end after a complete PI/OPD cycle")
        if self.model_name not in {"Qwen3-4B", "Qwen3-8B"} or self.task != "math":
            raise ValueError("The cyclic screen supports 4B/8B math only")
        if self.context != "worked" or self.target != "frozen" or self.micro_batch_size != 1:
            raise ValueError("Cyclic training requires worked/frozen launch context and microbatch one")
        if self.teacher_ema_decay is not None or self.ema_source is not None or self.replay_path is not None:
            raise ValueError("Cyclic training cannot combine EMA or replay")
        if self.save_checkpoints or self.retain_final_snapshot or self.stop_after_rollout is not None:
            raise ValueError("Cyclic jobs require complete cycles and temporary evaluation snapshots only")
        if self.evaluation_queue is None or self.snapshot_interval != 1:
            raise ValueError("Cyclic evaluation requires its queue and the per-update snapshot decision")


def execute(args: ScriptArgs):
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1" or not os.environ.get("RAY_ADDRESS"):
        raise ValueError("Create an isolated Ray cluster and set MILES_SCRIPT_EXTERNAL_RAY=1 and RAY_ADDRESS")
    if args.capture_full_update and (args.replay_path is None or args.target != "frozen"):
        raise ValueError("Full-update capture is only for a frozen-teacher correctness replay")
    model = shlex.quote(f"{args.model_dir}/{args.model_name}")
    initial = shlex.quote(args.initial_checkpoint) if args.initial_checkpoint else model
    checkpoint = (
        f"--hf-checkpoint {model} --load {initial} --opd-teacher-load {model} "
        "--megatron-to-hf-mode bridge --finetune --no-load-optim --no-load-rng "
        "--start-rollout-id 0 "
    )
    if args.save_checkpoints:
        checkpoint += f"--save {shlex.quote(f'{args.output_dir}/{args.run_id}/checkpoints')} "
    if args.save_checkpoints or args.evaluation_queue is not None:
        snapshot_dir = "hf" if args.save_checkpoints else "eval-snapshots"
        checkpoint += (
            f"--save-interval {args.snapshot_interval} "
            f"--save-hf {shlex.quote(f'{args.output_dir}/{args.run_id}/{snapshot_dir}/step_{{rollout_id}}')} "
        )
    if args.evaluation_queue is not None:
        checkpoint += (
            "--custom-megatron-post-save-hook-path miles.utils.opsd_study.enqueue_evaluation "
            f"--opsd-eval-queue {shlex.quote(args.evaluation_queue)} "
        )
        if args.retain_final_snapshot:
            checkpoint += "--opsd-retain-final-eval-snapshot "
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
        if args.capture_full_update:
            rollout += "--custom-megatron-before-train-step-hook-path tools.opsd.verify_full_update.capture_gradients "
    algorithm = (
        "--loss-type opsd_loss --disable-compute-advantages-and-returns "
        "--opsd-beta 0 --opsd-temperature 1.0 --opsd-token-clip 0 --opsd-reduction sample_mean "
        f"--opsd-context {args.context} --opsd-target {args.target} "
        f"--opsd-task {args.task} "
        f"--opsd-target-cache-gib 8 --opsd-target-microbatch-gib {2 * args.micro_batch_size} "
        "--lora-rank 0 "
    )
    if args.teacher_ema_decay is not None:
        algorithm += f"--opsd-teacher-ema-decay {args.teacher_ema_decay} "
    if args.ema_source is not None:
        algorithm += f"--opsd-ema-source {shlex.quote(args.ema_source)} "
    if args.cyclic_teacher_policy is not None:
        algorithm += (
            f"--opsd-cyclic-teacher-policy {args.cyclic_teacher_policy} "
            f"--opsd-cyclic-pi-updates {args.cyclic_pi_updates} "
            f"--opsd-cyclic-opd-updates {args.cyclic_opd_updates} "
            f"--opsd-cyclic-optimizer-policy {args.cyclic_optimizer_policy} "
            f"--opsd-cyclic-lr-schedule {args.cyclic_lr_schedule} "
        )
    lr_schedule = "--lr-decay-style constant "
    if args.cyclic_lr_schedule == "global_linear":
        # Megatron advances after each update: N-1 intervals put the floor on update N.
        lr_schedule = f"--lr-decay-style linear --min-lr 5e-7 --lr-decay-iters {args.num_rollout - 1} "
    optimizer = (
        f"--optimizer adam --lr 5e-6 {lr_schedule}--weight-decay 0 "
        "--adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 0.1 "
    )
    parallel = (
        "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--global-batch-size 4 --micro-batch-size {args.micro_batch_size} --seq-length 8192 "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node 2 --colocate --train-backend megatron "
    )
    inference = (
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 "
        "--sglang-enable-deterministic-inference --sglang-sampling-backend pytorch "
        "--sglang-cuda-graph-max-bs-decode 4 --sglang-max-running-requests 4 "
    )
    misc = (
        f"--seed {args.seed} --attention-dropout 0 --hidden-dropout 0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
    )
    if args.stop_after_rollout is not None:
        misc += f"--debug-exit-after-rollout {args.stop_after_rollout} "
    args.create_backend().execute_train(
        train_args=(
            checkpoint
            + rollout
            + algorithm
            + optimizer
            + parallel
            + inference
            + misc
            + U.get_default_wandb_args(__file__, run_id=args.run_id, opsd_profile=True)
        ).strip(),
        megatron_model_type=args.model_name.replace("Qwen", "qwen"),
        megatron_path=args.megatron_path,
        num_gpus_per_node=2,
        job_lifetime="launcher",
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
