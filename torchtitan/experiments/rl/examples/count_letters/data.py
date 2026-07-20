# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Self-contained dataset for the count_letters custom RL task (Tier 2 extension test).

No external/HF dependency: samples are generated from a fixed word list + RNG.
The model is shown a word and a target letter and must count occurrences.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from torchtitan.config import Configurable

_WORDS = [
    "strawberry", "banana", "mississippi", "apple", "cheese", "balloon",
    "committee", "raspberry", "tennessee", "bookkeeper", "assessment",
    "possesses", "parallel", "millennium", "necessary", "occurrence",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class CountLettersSample:
    """One count-the-letter problem: the word, the target letter, and the true count."""

    word: str
    letter: str
    expected_count: int


class CountLettersDataset(Configurable):
    """Yields (word, letter, expected_count) problems drawn from a fixed word list."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 1234

    def __init__(self, config: Config) -> None:
        self._rng = random.Random(config.seed)

    def __iter__(self) -> Iterator[CountLettersSample]:
        return self

    def __next__(self) -> CountLettersSample:
        word = self._rng.choice(_WORDS)
        # pick a letter that actually appears (so answers span 1..N, learnable)
        letter = self._rng.choice(sorted(set(word)))
        return CountLettersSample(
            word=word, letter=letter, expected_count=word.count(letter)
        )
