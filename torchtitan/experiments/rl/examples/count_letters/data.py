# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Self-contained dataset for the count_letters custom RL task (Tier 2 extension test).

No external/HF dependency. To AVOID reward saturation (a model can memorize a tiny
fixed word list in a few steps, which flatlines reward and collapses GRPO advantages
-> gradient blow-up), we generate a large, varied pool of longer pseudo-words so the
task keeps providing a learning signal for many steps.
"""
from __future__ import annotations

import random
import string
from collections.abc import Iterator
from dataclasses import dataclass

from torchtitan.config import Configurable

# A larger base vocabulary + procedural generation keeps the task from being memorized.
_BASE_WORDS = [
    "strawberry", "banana", "mississippi", "committee", "raspberry", "tennessee",
    "bookkeeper", "assessment", "possesses", "parallel", "millennium", "necessary",
    "occurrence", "accommodate", "embarrassment", "questionnaire", "unnecessarily",
    "committee", "aardvark", "bubblegum", "cellophane", "dependable", "effervescent",
]


def _make_word(rng: random.Random) -> str:
    """Either a base word or a longer procedurally-built one with repeated letters,
    so exact counts are non-trivial and the sample space is large."""
    if rng.random() < 0.4:
        return rng.choice(_BASE_WORDS)
    # build a 8-16 char pseudo-word with deliberately repeated letters
    length = rng.randint(8, 16)
    letters = rng.choices(string.ascii_lowercase[:12], k=length)  # bias to a-l -> repeats
    return "".join(letters)


@dataclass(frozen=True, kw_only=True, slots=True)
class CountLettersSample:
    """One count-the-letter problem: the word, the target letter, and the true count."""

    word: str
    letter: str
    expected_count: int


class CountLettersDataset(Configurable):
    """Yields (word, letter, expected_count) problems from a large procedural pool."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 1234

    def __init__(self, config: Config) -> None:
        self._rng = random.Random(config.seed)

    def __iter__(self) -> Iterator[CountLettersSample]:
        return self

    def __next__(self) -> CountLettersSample:
        word = _make_word(self._rng)
        # pick a letter that actually appears (so answers span 1..N, learnable)
        letter = self._rng.choice(sorted(set(word)))
        return CountLettersSample(
            word=word, letter=letter, expected_count=word.count(letter)
        )
