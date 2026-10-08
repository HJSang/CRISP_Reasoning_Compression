"""Tiny GPU check of the real Miles prepass and packed loss boundary.

Run from the Miles root in its GPU environment:
  torchrun --standalone --nproc-per-node=1 tests/manual/opsd/check_megatron.py --work-dir /tmp/opsd-check

Uses synthetic tokens and a random tiny Qwen3; downloads no model or dataset.
This verifies the scoring plumbing and frozen teacher, not paper reproduction.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.bridge.peft.canonical_lora import CanonicalLoRA
from megatron.core import parallel_state as mpu, tensor_parallel
from transformers import Qwen3Config, Qwen3ForCausalLM

from miles.backends.megatron_utils.checkpoint import _load_checkpoint_hf
from miles.backends.megatron_utils.opsd import score_teacher, verify_teacher_base, zero_teacher_adapters
from miles.backends.megatron_utils.model import run_forward_backward_pass
from miles.backends.megatron_utils.lora.utils import reduce_marked_lora_grads
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.tensor_backper import TensorBackuper
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase


def _initialize():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    mpu.initialize_model_parallel(tensor_model_parallel_size=size)
    tensor_parallel.model_parallel_cuda_manual_seed(41)
    trivial = GroupInfo(rank=0, size=1, group=None)
    tp = GroupInfo(rank=dist.get_rank(), size=size, group=mpu.get_tensor_model_parallel_group())
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=tp,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


def _models(work_dir):
    torch.manual_seed(41)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    reference = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    reference.config._attn_implementation = "eager"
    if dist.get_rank() == 0:
        reference.save_pretrained(work_dir / "base")
    dist.barrier()
    bridge = AutoBridge.from_hf_pretrained(str(work_dir / "base"), torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = dist.get_world_size()
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = False
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.hidden_dropout = 0.0
    provider.attention_dropout = 0.0
    provider.gradient_accumulation_fusion = False
    provider.calculate_per_token_loss = False
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    peft = CanonicalLoRA(dim=4, alpha=8, dropout=0.0)
    model = peft(model)
    # Exercise the same HF reload operation used by the actor teacher snapshot.
    _load_checkpoint_hf(
        model, None, SimpleNamespace(megatron_to_hf_mode="bridge", fp16=False, bf16=True), str(work_dir / "base")
    )
    return reference, model, peft


def _args(model):
    return SimpleNamespace(
        loss_type="opsd_loss",
        vocab_size=128,
        # Regression: Bridge may leave its 128-column head unpadded while the
        # launch arguments describe a 256-column padded vocabulary.
        padded_vocab_size=256,
        opsd_beta=0.0,
        opsd_temperature=1.1,
        opsd_token_clip=0.05,
        opsd_target_cache_gib=0.01,
        opsd_target_microbatch_gib=0.01,
        micro_batch_size=1,
        seq_length=256,
        qkv_format="thd",
        data_pad_size_multiplier=1,
        allgather_cp=False,
        use_rollout_entropy=False,
        enable_witness=False,
        custom_megatron_before_log_prob_hook_path=None,
        use_dynamic_batch_size=False,
        dumper_enable=False,
        dumper_fwd_only=[],
        dumper_fwd_bwd=[],
        use_sampling_support_replay=False,
        decoder_seq_length=None,
    )


def _data():
    return dict(
        tokens=[torch.tensor([1, 2, 3, 4, 5], device="cuda"), torch.tensor([4, 5, 6, 7], device="cuda")],
        teacher_prompt_ids=[[1, 9, 8, 2], [3, 2, 1]],
        response_lengths=[2, 1],
        total_lengths=[5, 4],
        loss_masks=[torch.ones(2, device="cuda"), torch.ones(1, device="cuda")],
        sample_indices=[7, 11],
    )


def _run(work_dir):
    reference, model, peft = _models(work_dir)
    args, data = _args(model), _data()
    backuper = TensorBackuper.create(lambda: model[0].named_parameters())
    backuper.backup("actor")
    zero_teacher_adapters(model)
    backuper.backup("teacher")
    verify_teacher_base(backuper)
    backuper.restore("actor")
    actor = SimpleNamespace(args=args, model=model, _switch_model=backuper.restore)
    score_teacher(actor, data, [2], rollout_id=0)
    initial_targets = [target.log_probs.clone() for target in data["opsd_targets"]]
    # Compare actual Megatron prepass distributions to an independent HF base.
    maximum_error = 0.0
    for target, prefix, tokens, length in zip(
        data["opsd_targets"], data["teacher_prompt_ids"], data["tokens"], data["response_lengths"], strict=True
    ):
        ids = torch.tensor([prefix + tokens[-length:].tolist()], device="cuda")
        with torch.no_grad():
            expected = (reference(ids).logits[0, len(prefix) - 1 : -1].float() / args.opsd_temperature).log_softmax(-1)
        start, width = target.vocab_start, target.log_probs.size(-1)
        expected = expected[:, start : start + width].cpu()
        error = (target.log_probs - expected).abs().max().item()
        maximum_error = max(maximum_error, error)
        torch.testing.assert_close(target.log_probs, expected, atol=0.02, rtol=0.005)

    parameters = [param for chunk in model for param in chunk.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-6, weight_decay=0.0)
    losses = []
    for step in range(2):
        iterator = DataIterator(data, micro_batch_size=1)
        optimizer.zero_grad()
        dumper = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=step)
        metrics = run_forward_backward_pass(args, dumper, [iterator], model, 2, num_rollouts=2)
        losses.extend(metric["values"][1].item() for metric in metrics)
        assert all(torch.isfinite(metric["values"]).all() for metric in metrics)
        reduce_marked_lora_grads(model)
        assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in parameters)
        assert all(param.grad is None for param in model[0].parameters() if not param.requires_grad)
        optimizer.step()
        backuper.backup("actor")
        score_teacher(actor, data, [2], rollout_id=step + 1)
        for before, after in zip(initial_targets, data["opsd_targets"], strict=True):
            torch.testing.assert_close(before, after.log_probs, atol=0, rtol=0)
    # Exercise restoration after failure inside forward_only, after eval mode and
    # teacher weights have been installed, not just an input-validation failure.
    actor_weights = {name: param.detach().clone() for name, param in model[0].named_parameters()}
    prior_modes = [module.training for module in model]
    args.custom_megatron_before_log_prob_hook_path = "builtins.len"
    try:
        score_teacher(actor, data, [2], rollout_id=9)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected deliberately invalid hook to fail")
    for name, param in model[0].named_parameters():
        torch.testing.assert_close(param, actor_weights[name], atol=0, rtol=0)
    assert [module.training for module in model] == prior_modes
    # Aggregate across TP so the public summary cannot hide a worse shard.
    error_tensor = torch.tensor(maximum_error, device="cuda")
    dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
    maximum_error = error_tensor.item()
    if dist.get_rank() == 0:
        result = dict(
            status="passed",
            tp=dist.get_world_size(),
            hf_teacher_max_abs_logprob_error=maximum_error,
            microbatch_losses=losses,
            optimizer_steps=2,
            frozen_teacher_bitwise_stable=True,
            exception_restore_passed=True,
            pipeline_backward=True,
        )
        (work_dir / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    _initialize()
    try:
        _run(args.work_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
