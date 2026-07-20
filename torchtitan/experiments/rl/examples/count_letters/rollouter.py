# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Trivial custom Rollouter for the count_letters task (Tier 2: extension surface).

Pure config: wires the dataset, env, and rubric. `make_env_group`, `get_*_sample`,
and `score_group` are inherited from the base `Rollouter` -- ZERO infra changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.examples.count_letters.data import CountLettersDataset
from torchtitan.experiments.rl.examples.count_letters.env import CountLettersEnv
from torchtitan.experiments.rl.examples.count_letters.rubric import RewardCountLetters
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rubrics import Rubric


class CountLettersRollouter(Rollouter):
    """Wires up the count_letters task: train/val datasets, env, and reward."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: CountLettersDataset.Config = field(
            default_factory=lambda: CountLettersDataset.Config(seed=42)
        )
        validation_dataset: CountLettersDataset.Config = field(
            default_factory=lambda: CountLettersDataset.Config(seed=99)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardCountLetters.Config(weight=1.0)]
            )
        )
        message_env: CountLettersEnv.Config = field(
            default_factory=CountLettersEnv.Config
        )
