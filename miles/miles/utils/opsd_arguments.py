"""Run-wide constraints for the first, frozen-base OPSD implementation."""

import math
from pathlib import Path


def add_opsd_arguments(parser):
    group = parser.add_argument_group("OPSD full-vocabulary loss")
    group.add_argument("--opsd-beta", type=float, default=0.0)
    group.add_argument(
        "--opsd-context", choices=["original", "none", "answer", "worked", "unrelated", "empty"], default="original"
    )
    group.add_argument("--opsd-target", choices=["frozen", "current", "transported"], default="frozen")
    group.add_argument("--opsd-reduction", choices=["reference", "sample_mean"], default="reference")
    group.add_argument(
        "--opsd-eval-queue",
        type=Path,
        default=None,
        help="Private checkpoint evaluation queue for the study save hook.",
    )
    group.add_argument(
        "--opsd-aggregate-staging-gib",
        type=float,
        default=24.0,
        help="Maximum aggregate CPU distribution storage per rank for transported targets.",
    )
    group.add_argument(
        "--opsd-temperature", type=float, default=1.1, help="Loss temperature, independent of sampling."
    )
    group.add_argument(
        "--opsd-token-clip", type=float, default=0.05, help="Clip each vocabulary contribution; 0 disables."
    )
    group.add_argument(
        "--opsd-target-cache-gib", type=float, default=8.0, help="Maximum CPU target storage per worker."
    )
    group.add_argument(
        "--opsd-target-microbatch-gib",
        type=float,
        default=1.0,
        help="Maximum target payload per microbatch; excludes activations and loss temporaries.",
    )
    return parser


def validate_opsd_args(args):
    if args.loss_type != "opsd_loss":
        return
    if not 0 <= args.opsd_beta <= 1:
        raise ValueError("--opsd-beta must be in [0, 1]")
    for name in (
        "opsd_temperature",
        "opsd_target_cache_gib",
        "opsd_target_microbatch_gib",
        "opsd_aggregate_staging_gib",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.opsd_token_clip) or args.opsd_token_clip < 0:
        raise ValueError("--opsd-token-clip must be finite and nonnegative")
    requirements = {
        "Megatron backend": args.train_backend == "megatron",
        "PP=1": getattr(args, "pipeline_model_parallel_size", 1) == 1,
        "CP=1": getattr(args, "context_parallel_size", 1) == 1,
        "THD packing": args.qkv_format == "thd" and not getattr(args, "allgather_cp", False),
        "no loss recomputation": not getattr(args, "recompute_loss_function", False),
        "canonical LoRA": args.lora_rank > 0 and args.lora_type == "canonical_lora",
        "single adapter": not args.multi_lora,
        "dense model": not getattr(args, "num_experts", None),
        "fixed microbatch schedule": not args.use_dynamic_batch_size and not args.use_dynamic_global_batch_size,
        "disabled advantage computation": not args.compute_advantages_and_returns,
        "no additional OPD/KL objective": not args.use_opd and not args.use_kl_loss and args.kl_coef == 0,
        "synchronous fresh single-turn rollout": not args.fully_async and not args.partial_rollout,
        "student adapter weight publication": not args.lora_train_only and args.update_weights_interval == 1,
        "raw problem input": not args.apply_chat_template,
        "OPSD generator": args.custom_generate_function_path == "miles.rollout.generate_hub.opsd.generate",
        "no routing/indexer replay": not args.use_rollout_routing_replay and not args.use_rollout_indexer_replay,
        "reference microbatch token mean": not args.calculate_per_token_loss,
        "no MTP auxiliary loss": not getattr(args, "mtp_num_layers", None),
    }
    for description, satisfied in requirements.items():
        if not satisfied:
            raise ValueError(f"OPSD v1 requires {description}")
    if args.opd_teacher_load is None or not Path(args.opd_teacher_load).is_dir():
        raise ValueError("OPSD requires --opd-teacher-load pointing to the HF or converted frozen base checkpoint")
    if args.opsd_context == "original" and args.opsd_target != "frozen":
        raise ValueError("Original compatibility mode requires the frozen teacher")
