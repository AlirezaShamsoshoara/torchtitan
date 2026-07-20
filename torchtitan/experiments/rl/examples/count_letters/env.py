# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Single-turn MessageEnv for the count_letters task."""
from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.count_letters.data import CountLettersSample


class CountLettersEnv(MessageEnv):
    """One-turn chat: ask the model to count a letter in a word and answer in an XML tag."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: CountLettersSample) -> None:
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        w = self._env_input.word
        c = self._env_input.letter
        prompt = (
            f"How many times does the letter '{c}' appear in the word \"{w}\"?\n\n"
            "Respond with ONLY the integer count inside this exact format:\n"
            "<count>N</count>\n"
            "where N is the number."
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": prompt}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        # Single-turn task: the assistant's first answer ends the rollout.
        return MessageEnvStepOutput(done=True)
