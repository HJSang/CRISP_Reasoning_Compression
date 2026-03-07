"""
Self-distillation worker: extends AsyncActorRolloutRefWorker with an SFT update step.

The worker reuses the full HybridEngine infrastructure (FSDP model, sglang rollout,
weight sync) and adds a single new registered method ``update_sft`` that performs
supervised fine-tuning using cross-entropy loss on verified correct responses.

This keeps the weight sync (rollout_mode / trainer_mode) and the generation path
identical to GRPO — we only replace the RL loss with SFT loss.
"""

import logging
from typing import Any

import psutil
import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.protocol import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.device import get_device_id
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__name__)


class SelfDistillWorker(AsyncActorRolloutRefWorker):
    """Worker that adds an SFT training step to the HybridEngine actor+rollout worker.

    Inherits all capabilities from AsyncActorRolloutRefWorker:
      - init_model, wake_up, sleep, chat_completion, generate
      - rollout_mode / trainer_mode for weight sync
      - FSDP model + optimizer + scheduler management

    Adds:
      - update_sft: SFT training with cross-entropy loss
    """

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_sft(self, data: DataProto) -> DataProto:
        """Perform one SFT training step on verified correct responses.

        The input DataProto should contain:
          - input_ids:      (B, max_length) — tokenized [prompt + response + padding]
          - attention_mask:  (B, max_length) — 1 for real tokens, 0 for padding
          - position_ids:    (B, max_length) — sequential position IDs
          - loss_mask:       (B, max_length) — 1 for response tokens (to train on), 0 otherwise

        The method:
          1. Loads FSDP model + optimizer to GPU (if offloaded)
          2. Splits data into micro-batches for gradient accumulation
          3. Computes cross-entropy loss masked by loss_mask
          4. Clips gradients and steps optimizer + scheduler
          5. Offloads model + optimizer back to CPU (if offloaded)

        Returns:
            DataProto with meta_info containing training metrics.
        """
        assert self._is_actor, "update_sft requires actor role"

        # --- Load model & optimizer to GPU if offloaded ---
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # will be moved per micro-batch
            metrics = self._sft_training_step(data)

            # LR scheduler step (same as update_actor)
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["sft/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() / (1024**3)
            )
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() / (1024**3)
            )
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        # --- Offload model & optimizer back to CPU ---
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_sft", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_sft", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_val_loss(self, data: DataProto) -> DataProto:
        """Compute validation loss (forward-only, no gradients).

        Uses the same forward+loss paths as update_sft but without backward
        pass, optimizer step, or gradient accumulation. The model is placed
        in eval mode for correct dropout/batchnorm behavior.

        Args:
            data: DataProto with input_ids, attention_mask, position_ids, loss_mask.

        Returns:
            DataProto with meta_info containing {"val_loss": float, "val_tokens": int}.
        """
        assert self._is_actor, "compute_val_loss requires actor role"

        # --- Load model to GPU if offloaded ---
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            metrics = self._val_forward(data)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        # --- Offload model back to CPU ---
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_val_loss", logger=logger)

        return output

    def _val_forward(self, data: DataProto) -> dict:
        """Forward-only loss computation over micro-batches (no backward)."""
        self.actor_module_fsdp.eval()

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )

        batch_size = data.batch["input_ids"].shape[0]
        if batch_size == 0:
            return {"val_loss": 0.0, "val_tokens": 0}

        micro_batches = data.split(micro_batch_size)
        device = get_device_id()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(device)
                input_ids = micro_batch.batch["input_ids"]
                attention_mask = micro_batch.batch["attention_mask"]
                position_ids = micro_batch.batch["position_ids"]
                loss_mask = micro_batch.batch["loss_mask"]

                if use_remove_padding:
                    loss, n_tokens = self._forward_loss_unpadded(
                        input_ids, attention_mask, position_ids, loss_mask
                    )
                else:
                    loss, n_tokens = self._forward_loss_padded(
                        input_ids, attention_mask, position_ids, loss_mask
                    )

                total_loss += loss.detach().item() * n_tokens.detach().item()
                total_tokens += n_tokens.detach().item()

        # Weighted average loss across all micro-batches
        avg_loss = total_loss / max(1, total_tokens)
        return {"val_loss": avg_loss, "val_tokens": int(total_tokens)}

    def _sft_training_step(self, data: DataProto) -> dict:
        """Core SFT training logic: micro-batch gradient accumulation with CE loss.

        Supports two modes:
          - **Padded** (use_remove_padding=false): Standard forward on right-padded sequences.
            Logits are (micro_B, max_len, vocab_size) — expensive for long max_len.
          - **Unpadded** (use_remove_padding=true): Removes padding before the forward pass
            using flash_attn_varlen. Logits are (1, total_nnz, vocab_size) where total_nnz
            is the sum of actual sequence lengths — typically 4–8× smaller than padded.

        The unpadded path mirrors VERL's ``DataParallelPPOActor._forward_micro_batch``
        and ``FSDPSFTTrainer._compute_loss_and_backward`` to ensure compatibility with
        the model's flash attention varlen patches.

        Args:
            data: DataProto with input_ids, attention_mask, position_ids, loss_mask.

        Returns:
            Dictionary of training metrics.
        """
        self.actor_module_fsdp.train()

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        # Determine micro-batch size from actor config
        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )

        # Split into micro-batches
        batch_size = data.batch["input_ids"].shape[0]
        if batch_size == 0:
            return {"sft/loss": 0.0, "sft/num_tokens": 0, "sft/batch_size": 0}

        micro_batches = data.split(micro_batch_size)
        n_micro_batches = len(micro_batches)
        grad_accum = max(1, n_micro_batches)

        device = get_device_id()

        self.actor_optimizer.zero_grad()
        total_loss = 0.0
        total_tokens = 0

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(device)
            input_ids = micro_batch.batch["input_ids"]
            attention_mask = micro_batch.batch["attention_mask"]
            position_ids = micro_batch.batch["position_ids"]
            loss_mask = micro_batch.batch["loss_mask"]  # (micro_B, max_len)

            if use_remove_padding:
                loss, n_tokens = self._forward_loss_unpadded(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            else:
                loss, n_tokens = self._forward_loss_padded(
                    input_ids, attention_mask, position_ids, loss_mask
                )

            # Scale for gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            total_loss += loss.detach().item()
            total_tokens += n_tokens.detach().item()

        # --- Gradient clipping and optimizer step ---
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        if isinstance(self.actor_module_fsdp, FSDP):
            grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
        else:
            from torch.distributed._composable.fsdp import FSDPModule
            if isinstance(self.actor_module_fsdp, FSDPModule):
                from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
                grad_norm = fsdp2_clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        else:
            logger.warning("Non-finite grad_norm (%.4f), skipping optimizer step", grad_norm.item())
            self.actor_optimizer.zero_grad()

        avg_loss = total_loss / max(1, n_micro_batches)

        return {
            "sft/loss": avg_loss,
            "sft/grad_norm": grad_norm.detach().item(),
            "sft/num_tokens": int(total_tokens),
            "sft/batch_size": batch_size,
        }

    # ------------------------------------------------------------------
    # Forward + Loss: Padded path (original)
    # ------------------------------------------------------------------

    def _forward_loss_padded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard padded forward pass + CE loss.

        Logits shape: (micro_B, max_len, vocab_size). Memory-intensive for long
        max_len because every sequence is padded to the full sft_max_length.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.actor_module_fsdp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits
            del outputs

        vocab_size = logits.shape[-1]
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous()

        flat_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        del logits

        flat_labels = shift_labels.view(-1)
        flat_mask = shift_loss_mask.view(-1)

        ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        del flat_logits
        masked_loss = ce_loss * flat_mask
        n_tokens = flat_mask.sum().clamp(min=1)
        loss = masked_loss.sum() / n_tokens

        return loss, n_tokens

    # ------------------------------------------------------------------
    # Forward + Loss: Unpadded path (remove_padding)
    # ------------------------------------------------------------------

    def _forward_loss_unpadded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpadded forward pass + CE loss using flash_attn_varlen.

        Strips padding tokens before the forward pass so the logits tensor is
        (1, total_nnz, vocab_size) where total_nnz = sum of actual sequence
        lengths.  For typical math/reasoning sequences (~4K tokens padded to
        32K), this provides a **4–8× memory reduction** on logits.

        The approach mirrors VERL's ``DataParallelPPOActor._forward_micro_batch``
        for unpadding and ``FSDPSFTTrainer._compute_loss_and_backward`` for the
        shifted CE loss on packed sequences.
        """
        # --- Unpad inputs ---
        # unpad_input expects (B, S, D); we unsqueeze input_ids to add a dummy D=1 dim.
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad: (total_nnz, 1)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # Unpad position_ids to keep rotary embeddings aligned
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)  # (1, total_nnz)

        # --- Forward pass with attention_mask=None → flash_attn_varlen ---
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.actor_module_fsdp(
                input_ids=input_ids_rmpad,
                attention_mask=None,  # triggers flash_attn_varlen
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)  # (total_nnz, vocab_size)
            del outputs

        # --- Shifted labels for next-token prediction on packed sequences ---
        # torch.roll shifts all tokens left by 1.  At sequence boundaries the
        # label wraps incorrectly, but loss_mask is 0 at those positions so
        # the gradient contribution is zero — identical to VERL's approach.
        input_ids_flat = input_ids_rmpad.squeeze(0)  # (total_nnz,)
        shifted_labels = torch.roll(input_ids_flat, shifts=-1, dims=0)

        # Unpad loss_mask using the same indices, then shift
        loss_mask_flat = loss_mask.reshape(-1)  # (B * max_len,)
        loss_mask_rmpad = loss_mask_flat[indices]  # (total_nnz,)
        # shifted_loss_mask[i] = loss_mask for position i+1 (the target token).
        # The last position has no valid target → 0.
        shifted_loss_mask = torch.zeros_like(loss_mask_rmpad)
        shifted_loss_mask[:-1] = loss_mask_rmpad[1:]

        # --- Cross-entropy loss ---
        ce_loss = F.cross_entropy(logits_rmpad, shifted_labels, reduction="none")
        del logits_rmpad
        masked_loss = ce_loss * shifted_loss_mask
        n_tokens = shifted_loss_mask.sum().clamp(min=1)
        loss = masked_loss.sum() / n_tokens

        return loss, n_tokens
