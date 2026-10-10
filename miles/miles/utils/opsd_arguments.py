"""Run-wide constraints for OPSD scoring and teacher updates."""

import math
from pathlib import Path

from miles.utils.environ import use_legacy_rollout_v1


def add_opsd_arguments(parser):
    group = parser.add_argument_group("OPSD full-vocabulary loss")
    group.add_argument("--opsd-beta", type=float, default=0.0)
    group.add_argument(
        "--opsd-context", choices=["original", "none", "answer", "worked", "unrelated", "empty"], default="original"
    )
    group.add_argument("--opsd-target", choices=["frozen", "current", "transported"], default="frozen")
    group.add_argument("--opsd-cyclic-teacher-policy", choices=["fixed_original", "cycle_refresh"], default=None,
                       help="Alternate worked PI and no-PI OPD with fresh Adam at every phase boundary.")
    group.add_argument("--opsd-cyclic-pi-updates", type=int, default=7)
    group.add_argument("--opsd-cyclic-opd-updates", type=int, default=4)
    group.add_argument("--opsd-teacher-ema-decay", type=float, default=None,
                       help="Optional FP32 teacher EMA after each successful optimizer step.")
    group.add_argument("--opsd-ema-source", type=Path, default=None,
                       help="Temporary FP32 teacher source paired with the starting student checkpoint.")
    group.add_argument("--opsd-task", choices=["math", "code"], default="math")
    group.add_argument("--opsd-reduction", choices=["reference", "sample_mean"], default="reference")
    group.add_argument(
        "--opsd-eval-queue",
        type=Path,
        default=None,
        help="Private checkpoint evaluation queue for the study save hook.",
    )
    group.add_argument(
        "--opsd-retain-final-eval-snapshot", action="store_true",
        help="Retain the evaluated final HF source until paired branches consume it; no optimizer save.",
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
        if getattr(args, "opsd_cyclic_teacher_policy", None) is not None:
            raise ValueError("Cyclic OPSD requires the opsd_loss objective")
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
        "full parameters or canonical LoRA": (args.lora_rank == 0 and getattr(args, "lora_adapter_path", None) is None)
        or (args.lora_rank > 0 and args.lora_type == "canonical_lora"),
        "single adapter": not args.multi_lora,
        "dense model": not getattr(args, "num_experts", None),
        "fixed microbatch schedule": not args.use_dynamic_batch_size and not args.use_dynamic_global_batch_size,
        "disabled advantage computation": not args.compute_advantages_and_returns,
        "no additional OPD/KL objective": not args.use_opd and not args.use_kl_loss and args.kl_coef == 0,
        "synchronous fresh single-turn rollout": not args.fully_async and not args.partial_rollout,
        "student weight publication": not args.lora_train_only and args.update_weights_interval == 1,
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
    decay = getattr(args, "opsd_teacher_ema_decay", None)
    if getattr(args, "opsd_task", "math") == "code" and args.opsd_context not in {"worked", "none"}:
        raise ValueError("Code OPSD supports worked reference code or no PI")
    if getattr(args, "opsd_ema_source", None) is not None and decay is None:
        raise ValueError("An EMA source requires an EMA teacher")
    if decay is not None:
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        if args.opsd_target != "frozen" or args.opsd_context != "worked" or args.lora_rank != 0:
            raise ValueError("EMA requires full-parameter direct worked-PI targets")
        if getattr(args, "debug_disable_optimizer", False) or not getattr(args, "check_for_nan_in_loss_and_grad", True):
            raise ValueError("EMA requires successful checked optimizer updates")
        if getattr(args, "use_fault_tolerance", False) or getattr(args, "indep_dp", False):
            raise ValueError("EMA recovery is not supported; restart from an explicit paired source")
    if getattr(args, "opsd_cyclic_teacher_policy", None) is not None:
        _validate_cyclic_args(args)


def _validate_cyclic_args(args):
    pi, opd = args.opsd_cyclic_pi_updates, args.opsd_cyclic_opd_updates
    if min(pi, opd) < 1 or args.num_rollout < 1 or args.num_rollout % (pi + opd):
        raise ValueError("Cyclic OPSD requires positive phase lengths and complete planned cycles")
    requirements = {
        "standard synchronous rollout with an explicit rollout ID":
            not use_legacy_rollout_v1() and getattr(args, "rollout_function_path", None) in (
                None, "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
            ),
        "full-parameter frozen direct targets starting in worked PI":
            args.lora_rank == 0 and args.opsd_target == "frozen" and args.opsd_context == "worked",
        "sample-mean forward KL at scoring temperature one without token clipping":
            args.opsd_reduction == "sample_mean" and args.opsd_beta == 0
            and args.opsd_temperature == 1 and args.opsd_token_clip == 0,
        "a separate cyclic policy without EMA": args.opsd_teacher_ema_decay is None,
        "one optimizer update per fresh rollout":
            args.rollout_batch_size * args.n_samples_per_prompt == args.global_batch_size,
        "checked Adam updates with constant LR and no warmup":
            args.optimizer == "adam" and args.lr_decay_style == "constant"
            and not getattr(args, "lr_warmup_iters", 0) and not getattr(args, "lr_warmup_fraction", None)
            and not getattr(args, "debug_disable_optimizer", False)
            and getattr(args, "check_for_nan_in_loss_and_grad", True),
        "fresh optimizer and rollout zero":
            getattr(args, "no_load_optim", False) and getattr(args, "start_rollout_id", None) == 0,
        "no recovery or per-update optimizer reset":
            not any(getattr(args, key, False) for key in (
                "use_fault_tolerance", "indep_dp", "reset_optimizer_states", "fp16",
                "offload_optimizer_states", "optimizer_cpu_offload", "stream_optimizer_state_to_disk",
                "chunked_optimizer_state_offload", "use_precision_aware_optimizer",
                "rematerialize_param_from_master_weight",
            )),
        "temporary evaluation snapshots only": args.save is None,
    }
    for description, satisfied in requirements.items():
        if not satisfied:
            raise ValueError(f"Cyclic OPSD requires {description}")
