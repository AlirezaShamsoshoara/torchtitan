# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.llama3 import model_registry as llama3_model_registry
from torchtitan.trainer import Trainer

from . import model_registry


def attn_res_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def _1b_trainer_config(dataset: str = "c4_test") -> Trainer.Config:
    """Shared training config for 1B model comparison (Llama3 vs AttnRes).

    Uses the Llama3 tokenizer, 8-way FSDP, and training hyperparameters
    appropriate for a 1.2B parameter model on 8x H100 GPUs.

    Args:
        dataset: Dataset name. "c4_test" (local subset) or "c4" (full streamed).
    """
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        # model_spec is set by the caller
        model_spec=None,  # type: ignore[arg-type]
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset=dataset,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        validator=Validator.Config(
            freq=0,
        ),
    )


def attn_res_1b() -> Trainer.Config:
    """AttnRes 1B trainer config for comparison with Llama3 1B."""
    config = _1b_trainer_config()
    config.model_spec = model_registry("1B")
    return config


def llama3_1b_baseline() -> Trainer.Config:
    """Llama3 1B baseline trainer config (for AttnRes comparison).

    This config lives in the AttnRes experiment to avoid modifying core
    Llama3 code. Run via: --module attn_residuals --config llama3_1b_baseline
    """
    config = _1b_trainer_config()
    config.model_spec = llama3_model_registry("1B")
    return config


def attn_res_1b_c4() -> Trainer.Config:
    """AttnRes 1B trainer config using the full C4 dataset."""
    config = _1b_trainer_config(dataset="c4")
    config.model_spec = model_registry("1B")
    return config


def llama3_1b_baseline_c4() -> Trainer.Config:
    """Llama3 1B baseline trainer config using the full C4 dataset.

    Run via: --module attn_residuals --config llama3_1b_baseline_c4
    """
    config = _1b_trainer_config(dataset="c4")
    config.model_spec = llama3_model_registry("1B")
    return config


def _8b_trainer_config() -> Trainer.Config:
    """Shared training config for 8B model comparison (Llama3 vs AttnRes).

    Uses the Llama3 tokenizer, 8-way FSDP, seq_len=8192, and training
    hyperparameters matching the upstream Llama3 8B config. Designed for
    8x H100 GPUs on a single node.
    """
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        model_spec=None,  # type: ignore[arg-type]
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=5000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        validator=Validator.Config(
            freq=0,
        ),
    )


def attn_res_8b() -> Trainer.Config:
    """AttnRes 8B trainer config for comparison with Llama3 8B."""
    config = _8b_trainer_config()
    config.model_spec = model_registry("8B")
    return config


def llama3_8b_baseline() -> Trainer.Config:
    """Llama3 8B baseline trainer config (for AttnRes comparison).

    This config lives in the AttnRes experiment to avoid modifying core
    Llama3 code. Run via: --module attn_residuals --config llama3_8b_baseline
    """
    config = _8b_trainer_config()
    config.model_spec = llama3_model_registry("8B")
    return config
