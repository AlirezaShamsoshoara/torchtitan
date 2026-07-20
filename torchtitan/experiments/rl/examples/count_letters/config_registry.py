# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Config registry for the count_letters custom task (Tier 2 extension test).

Demonstrates the "zero infra changes" claim: we reuse the alphabet_sort recipe
(model, parallelism, loss, sampling) verbatim and swap ONLY the rollouter to our
custom task. Discoverable via `--module count_letters --config <fn>`.
"""
from __future__ import annotations

import dataclasses

from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
    rl_grpo_qwen3_0_6b_varlen as _alphabet_sort_varlen,
    rl_grpo_qwen3_0_6b_varlen_batch_invariant as _alphabet_sort_bi,
)
from torchtitan.experiments.rl.examples.count_letters import CountLettersRollouter
from torchtitan.experiments.rl.examples.count_letters.rubric import RewardCountLetters
from torchtitan.experiments.rl.rubrics import Rubric


def rl_grpo_qwen3_0_6b_count_letters() -> Controller.Config:
    """Qwen3-0.6B GRPO on count_letters. Same recipe as alphabet_sort varlen,
    only the rollouter (task logic) differs -> proves the extension surface."""
    config = _alphabet_sort_varlen()
    config.rollouter = CountLettersRollouter.Config()
    return config


def rl_grpo_qwen3_0_6b_count_letters_format_only() -> Controller.Config:
    """Same task, but reward = format bonus ONLY (format_weight=1.0).

    Used to demonstrate a custom Rubric shapes the gradient: with format-only
    reward the model should learn to emit the <count> tag but NOT to count
    correctly, so exact-count accuracy should stay flat while format compliance
    rises -- visibly different learning signal from the default reward."""
    config = _alphabet_sort_varlen()
    config.rollouter = CountLettersRollouter.Config(
        rubric=Rubric.Config(
            reward_fns=[RewardCountLetters.Config(weight=1.0, format_weight=1.0)]
        )
    )
    return config


def rl_grpo_qwen3_0_6b_count_letters_batch_invariant() -> Controller.Config:
    """Batch-invariant variant of count_letters (8 GPUs), for parity-under-custom-task."""
    config = _alphabet_sort_bi()
    config.rollouter = CountLettersRollouter.Config()
    return config


def rl_grpo_qwen3_0_6b_count_letters_dapo() -> Controller.Config:
    """count_letters but with DAPO loss (asymmetric clip-higher 0.2/0.28) instead of GRPO.
    Demonstrates the GRPO<->DAPO swap changes behavior as documented (config-only)."""
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.experiments.rl.losses import DAPOLoss

    config = rl_grpo_qwen3_0_6b_count_letters()
    config.trainer.loss = ChunkedLossWrapper.Config(
        num_chunks=8,
        loss_fn=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
    )
    return config


def rl_grpo_qwen3_0_6b_count_letters_thinking() -> Controller.Config:
    """count_letters with Renderer thinking ON (model emits reasoning). Config-only toggle."""
    import dataclasses

    config = rl_grpo_qwen3_0_6b_count_letters()
    config.renderer = dataclasses.replace(config.renderer, enable_thinking=True)
    return config
