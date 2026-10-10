"""Reject launch modes that would silently change the validated OPSD objective."""

from argparse import ArgumentParser, Namespace

import pytest

from miles.utils.opsd_arguments import add_opsd_arguments, validate_opsd_args


def _args(tmp_path):
    parser = add_opsd_arguments(ArgumentParser())
    args = Namespace(
        **vars(parser.parse_args([])),
        loss_type="opsd_loss",
        train_backend="megatron",
        qkv_format="thd",
        lora_rank=4,
        lora_type="canonical_lora",
        multi_lora=False,
        use_dynamic_batch_size=False,
        use_dynamic_global_batch_size=False,
        compute_advantages_and_returns=False,
        use_opd=False,
        use_kl_loss=False,
        kl_coef=0,
        fully_async=False,
        partial_rollout=False,
        lora_train_only=False,
        update_weights_interval=1,
        apply_chat_template=False,
        custom_generate_function_path="miles.rollout.generate_hub.opsd.generate",
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        calculate_per_token_loss=False,
        opd_teacher_load=str(tmp_path),
    )
    return args


def test_default_opsd_contract(tmp_path):
    args = _args(tmp_path)
    validate_opsd_args(args)
    assert (args.opsd_beta, args.opsd_temperature, args.opsd_token_clip) == (0.0, 1.1, 0.05)


def test_full_parameter_study_accepts_zero_rank_but_rejects_adapter_load(tmp_path):
    args = _args(tmp_path)
    args.lora_rank = 0
    args.opsd_context = "worked"
    validate_opsd_args(args)
    args.lora_adapter_path = "adapter"
    with pytest.raises(ValueError, match="full parameters or canonical LoRA"):
        validate_opsd_args(args)


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("use_opd", True, "additional OPD"),
        ("calculate_per_token_loss", True, "reference microbatch"),
        ("apply_chat_template", True, "raw problem"),
        ("context_parallel_size", 2, "CP=1"),
        ("lora_type", "lora", "canonical"),
        ("opsd_temperature", float("nan"), "finite and positive"),
        ("opsd_target_cache_gib", 0, "finite and positive"),
    ],
)
def test_incompatible_mode_fails_before_scoring(tmp_path, key, value, match):
    args = _args(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError, match=match):
        validate_opsd_args(args)


@pytest.mark.parametrize("key,value", [("opsd_context", "none"), ("opsd_target", "transported"),
                                      ("lora_rank", 4), ("use_fault_tolerance", True),
                                      ("debug_disable_optimizer", True), ("opsd_teacher_ema_decay", float("nan"))])
def test_ema_rejects_ambiguous_or_unrecoverable_teacher_policy(tmp_path, key, value):
    args = _args(tmp_path)
    args.lora_rank = 0
    args.opsd_context = "worked"
    args.opsd_teacher_ema_decay = 0.9
    validate_opsd_args(args)
    setattr(args, key, value)
    with pytest.raises(ValueError, match="EMA"):
        validate_opsd_args(args)


def _cyclic_args(tmp_path):
    args = _args(tmp_path)
    args.__dict__.update(
        lora_rank=0, opsd_context="worked", opsd_reduction="sample_mean", opsd_temperature=1,
        opsd_token_clip=0, opsd_cyclic_teacher_policy="cycle_refresh", num_rollout=88,
        rollout_batch_size=4, n_samples_per_prompt=1, global_batch_size=4, optimizer="adam",
        lr_decay_style="constant", no_load_optim=True, start_rollout_id=0, save=None,
    )
    return args


@pytest.mark.parametrize("policy", ["fixed_original", "cycle_refresh"])
def test_cyclic_contract_accepts_both_teacher_sources(tmp_path, policy):
    args = _cyclic_args(tmp_path)
    args.opsd_cyclic_teacher_policy = policy
    validate_opsd_args(args)


@pytest.mark.parametrize("key,value", [
    ("loss_type", "policy_loss"),
    ("rollout_function_path", "custom.unidentified_rollout"),
    ("opsd_cyclic_pi_updates", 0), ("opsd_cyclic_opd_updates", -1), ("num_rollout", 87),
    ("lora_rank", 4), ("opsd_context", "none"), ("opsd_target", "current"),
    ("opsd_reduction", "reference"), ("opsd_beta", 0.5), ("opsd_temperature", 1.1),
    ("opsd_token_clip", 0.05), ("opsd_teacher_ema_decay", 0.9), ("global_batch_size", 8),
    ("optimizer", "sgd"), ("lr_decay_style", "linear"), ("lr_warmup_iters", 1),
    ("debug_disable_optimizer", True), ("check_for_nan_in_loss_and_grad", False),
    ("no_load_optim", False), ("start_rollout_id", 11), ("reset_optimizer_states", True),
    ("use_fault_tolerance", True), ("indep_dp", True), ("fp16", True),
    ("offload_optimizer_states", True), ("optimizer_cpu_offload", True),
    ("stream_optimizer_state_to_disk", True), ("chunked_optimizer_state_offload", True),
    ("use_precision_aware_optimizer", True), ("rematerialize_param_from_master_weight", True),
    ("save", "recovery"),
])
def test_cyclic_rejects_ambiguous_objective_reset_or_source(tmp_path, key, value):
    args = _cyclic_args(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError, match="Cyclic OPSD"):
        validate_opsd_args(args)


def test_cyclic_rejects_legacy_rollout_without_update_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "1")
    with pytest.raises(ValueError, match="explicit rollout ID"):
        validate_opsd_args(_cyclic_args(tmp_path))
