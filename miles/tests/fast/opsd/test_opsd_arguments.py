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
