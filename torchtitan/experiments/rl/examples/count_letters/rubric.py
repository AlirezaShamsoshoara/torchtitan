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
    """Reward in [0,1]. Strict EXACT-match on the count + small format bonus + a
    BREVITY term that penalizes long completions.

    Anti-saturation (G11): exact-match (0/1) keeps reward discriminating so GRPO
    advantages don't collapse to noise.

    Anti-length-inflation (G12): the earlier version had NO length term, so once the
    model could count, reward saturated (~0.90) and the policy drifted to ever-longer
    completions toward the max_tokens cap at zero reward cost -> generator decode time
    grew ~200x and the async loop crawled. This version multiplies the correctness
    reward by a brevity factor: full credit at/under `target_len` completion tokens,
    decaying linearly to `brevity_floor` by `max_len` tokens. The answer only needs a
    short "<count>N</count>", so concise correct answers score highest and rambling is
    penalized -- removing the incentive to inflate length.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        format_weight: float = 0.1
        """Fraction of reward given just for producing a parseable <count> tag."""
        target_len: int = 24
        """Completion token budget for full brevity credit (a short answer fits easily)."""
        max_len: int = 256
        """At/above this many completion tokens, brevity multiplier hits `brevity_floor`."""
        brevity_floor: float = 0.2
        """Minimum brevity multiplier for very long completions (keeps some signal)."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.format_weight = config.format_weight
        self.target_len = config.target_len
        self.max_len = config.max_len
        self.brevity_floor = config.brevity_floor

    def _brevity(self, n_tokens: int) -> float:
        """1.0 at/under target_len, linear decay to brevity_floor by max_len."""
        if n_tokens <= self.target_len:
            return 1.0
        if n_tokens >= self.max_len:
            return self.brevity_floor
        frac = (n_tokens - self.target_len) / (self.max_len - self.target_len)
        return 1.0 - frac * (1.0 - self.brevity_floor)

    async def __call__(self, rollout: Rollout, env_input: CountLettersSample) -> float:
        turn = rollout.turns[0] if rollout.turns else None
        msg = turn.completion_message if turn else None
        text = (msg.get("content") or "") if msg else ""
        n_tokens = len(turn.completion_token_ids) if turn else 0
        m = _COUNT_RE.search(text)
        if m is None:
            return 0.0
        try:
            guess = int(m.group(1))
        except ValueError:
            return 0.0
        fmt = self.format_weight  # parseable tag present
        correct = 1.0 if guess == env_input.expected_count else 0.0
        base = fmt + (1.0 - fmt) * correct
        # brevity gates the whole reward so long rollouts (even correct) score lower
        return base * self._brevity(n_tokens)
