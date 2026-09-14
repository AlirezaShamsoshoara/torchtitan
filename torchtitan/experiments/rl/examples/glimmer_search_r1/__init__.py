# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer Search-R1 RL example.

Runs the ``search_r1`` multi-turn retrieval-QA recipe on Muse Glimmer 30B. This
package supplies only the Glimmer-specific glue:

* ``config_registry.rl_muse_glimmer_search_r1`` — the RL config (model spec,
  renderer, TP=2 generator, FullAC trainer).
* ``renderer.MuseGlimmerRenderer`` — harmony chat format + ATEM tool calls,
  registered into the ``renderers`` library.
* ``tools/convert_qk_layout.py`` — mandatory one-time offline Q/K RoPE
  conversion of the released HF checkpoint.
* ``tools/smoke_generate.py`` — a fast generation sanity check.
* ``standin_retriever.py`` — a synthetic ``/retrieve`` server for POC runs.

The dataset / env / rubric are imported directly from ``search_r1`` at config
build time (not re-exported here) to avoid pulling optional deps at import.
"""

__all__: list[str] = []
