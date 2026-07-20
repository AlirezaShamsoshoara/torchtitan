# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Custom reward for count_letters: exact-match with partial credit by closeness."""
from __future__ import annotations

import re
from dataclasses import dataclass

from torchtitan.experiments.rl.examples.count_letters.data import CountLettersSample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn

_COUNT_RE = re.compile(r"<count>\s*(-?\d+)\s*</count>", re.IGNORECASE)


class RewardCountLetters(RewardFn):
    """Reward in [0,1]. 1.0 for exact count; partial credit that decays with |error|;
    small format bonus for emitting a well-formed <count> tag even if wrong.

    Shaping is deliberately simple so we can *see* the reward shape the gradient:
    reward = format_weight * has_tag + (1 - format_weight) * closeness.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        format_weight: float = 0.2
        """Fraction of reward given just for producing a parseable <count> tag."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.format_weight = config.format_weight

    async def __call__(self, rollout: Rollout, env_input: CountLettersSample) -> float:
        msg = rollout.turns[0].completion_message if rollout.turns else None
        text = (msg.get("content") or "") if msg else ""
        m = _COUNT_RE.search(text)
        if m is None:
            return 0.0
        try:
            guess = int(m.group(1))
        except ValueError:
            return 0.0
        fmt = self.format_weight  # parseable tag present
        err = abs(guess - env_input.expected_count)
        closeness = 1.0 if err == 0 else max(0.0, 1.0 - err / 5.0)
        return fmt + (1.0 - fmt) * closeness
