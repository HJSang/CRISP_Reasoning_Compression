"""
Main entry point for OPSD (On-Policy Self-Distillation) training with VERL's HybridEngine.

Uses Hydra for configuration management. Initializes Ray, creates workers,
dataset, and launches the OPSDTrainer.

Key differences from main_self_distill.py:
  - Uses OPSDWorker instead of SelfDistillWorker
  - Uses Role.ActorRolloutRef instead of Role.ActorRollout (ref model needed for teacher)
  - Uses OPSDTrainer instead of SelfDistillTrainer
  - Hydra config name: opsd_trainer instead of sd_trainer

Usage:
    python -m self_distill_hybrid.main_opsd \\
        --config-path ./config \\
        --config-name opsd_trainer \\
        actor_rollout_ref.model.path=/path/to/model \\
        data.train_files=/path/to/sd_prompts.parquet
"""

import logging
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="opsd_trainer", version_base=None)
def main(config):
    """Main entry point for OPSD training with Hydra configuration."""
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    run_opsd(config)


def run_opsd(config) -> None:
    """Initialize Ray cluster and run OPSD training.

    Args:
        config: Hydra/OmegaConf training configuration.
    """
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info("Ray init kwargs: %s", ray_init_kwargs)
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(OPSDTaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class OPSDTaskRunner:
    """Ray remote class for executing OPSD training.

    Key differences from SDTaskRunner:
      - Uses OPSDWorker (has update_opsd JSD training method)
      - Maps to Role.ActorRolloutRef (not Role.ActorRollout) so ref model is loaded
      - Uses OPSDTrainer instead of SelfDistillTrainer
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add OPSD worker (actor + rollout + ref with JSD update capability)."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        strategy = config.actor_rollout_ref.actor.strategy
        if strategy in {"fsdp", "fsdp2"}:
            from self_distill_hybrid.opsd_worker import OPSDWorker

            actor_rollout_cls = OPSDWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {strategy} not supported for OPSD")

        # Use ActorRolloutRef so the ref model (frozen teacher) is loaded
        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRolloutRef] = "global_pool"
        return ray_worker_group_cls

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager (single global pool)."""
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        return ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

    def run(self, config):
        """Execute the OPSD training workflow.

        Steps:
          1. Resolve config and print
          2. Create OPSD worker (actor + rollout + ref + opsd_update)
          3. Load tokenizer
          4. Create SelfDistillDataset from SD prompts parquet
          5. Initialize OPSDTrainer
          6. Init workers and start training
        """
        from verl.utils.fs import copy_to_local

        logger.info("OPSDTaskRunner hostname: %s, PID: %d", socket.gethostname(), os.getpid())
        try:
            OmegaConf.resolve(config)
            logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))
        except Exception as e:
            logger.warning("Could not fully resolve config: %s", e)
            logger.info("Unresolved config:\n%s", OmegaConf.to_yaml(config))

        # 1. Create worker
        ray_worker_group_cls = self.add_actor_rollout_worker(config)

        # 2. Load tokenizer
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 3. Create SD dataset (same format as self-distillation)
        from self_distill_hybrid.sd_dataset import SelfDistillDataset, collate_fn

        train_dataset = SelfDistillDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            config=config.data,
            max_samples=config.data.get("train_max_samples", -1),
        )

        # 4. Resolve SD val parquet path (for _compute_val_loss)
        sd_val_data_path = config.data.get("sd_val_files", None)
        if not sd_val_data_path:
            train_path = config.data.train_files
            if isinstance(train_path, str):
                val_candidate = train_path.replace(
                    "self_distill_prompts.parquet", "self_distill_prompts_val.parquet"
                )
                if val_candidate != train_path and os.path.exists(val_candidate):
                    sd_val_data_path = val_candidate
                    logger.info("Auto-detected SD val data: %s", sd_val_data_path)

        # 5. Load RL-format val dataset and reward manager for _validate()
        val_reward_fn = None
        val_dataset = None
        rl_val_files = config.data.get("val_files", None)
        if rl_val_files:
            from verl.trainer.ppo.reward import load_reward_manager

            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1,
                **config.reward_model.get("reward_kwargs", {}),
            )
            logger.info("Loaded val_reward_fn for _validate() (data_source-based routing)")

            from verl.trainer.main_ppo import create_rl_dataset

            val_dataset = create_rl_dataset(
                rl_val_files, config.data, tokenizer, processor,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )
            logger.info("Loaded RL val dataset: %d samples from %s", len(val_dataset), rl_val_files)

        # 6. Initialize resource pools
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # 7. Create and run OPSD trainer
        from self_distill_hybrid.opsd_trainer import OPSDTrainer

        trainer = OPSDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            collate_fn=collate_fn,
            val_data_path=sd_val_data_path,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
