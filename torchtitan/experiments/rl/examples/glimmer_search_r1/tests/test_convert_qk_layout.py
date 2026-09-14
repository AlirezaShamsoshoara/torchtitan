# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the offline Q/K RoPE layout conversion tool.

The tool's ``reverse_permute`` must be **bit-for-bit identical** to
``MuseGlimmerStateDictAdapter._reverse_permute`` (HF split-half -> torchtitan
interleaved), so that converting offline + loading with the adapter permute
disabled is exactly equivalent to loading unconverted with the adapter permute
enabled. This test asserts that equivalence on synthetic tensors at the real 30B
head geometry, and (when present) against a real q/k tensor from the released
checkpoint.

Run:
    python -m pytest torchtitan/experiments/rl/examples/glimmer_search_r1/tests/test_convert_qk_layout.py -v
or as a script:
    python torchtitan/experiments/rl/examples/glimmer_search_r1/tests/test_convert_qk_layout.py
"""

from __future__ import annotations

import importlib.util
import os

import torch

_TOOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "tools", "convert_qk_layout.py"
)


def _load_tool():
    # Import the tool file directly so we don't pull the rl package __init__
    # (which imports vllm) into a pure-CPU test.
    spec = importlib.util.spec_from_file_location("convert_qk_layout", _TOOL_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# reference math (copied verbatim from MuseGlimmerStateDictAdapter)
def _adapter_reverse_permute(w, nh):
    d1, d2 = w.shape
    return w.view(nh, 2, d1 // nh // 2, d2).transpose(1, 2).reshape(d1, d2).clone()


def _adapter_permute(w, nh):  # titan -> HF (forward)
    d1, d2 = w.shape
    return w.view(nh, d1 // nh // 2, 2, d2).transpose(1, 2).reshape(d1, d2).clone()


# real 30B text geometry
_N_HEADS, _N_KV, _HEAD_DIM, _DIM = 32, 2, 128, 6656


def test_bit_exact_vs_adapter_and_roundtrip():
    tool = _load_tool()
    torch.manual_seed(0)
    for name, nh in [("q", _N_HEADS), ("k", _N_KV)]:
        w = torch.randn(nh * _HEAD_DIM, _DIM, dtype=torch.float32)
        # bit-exact vs the adapter's reverse_permute
        assert torch.equal(tool.reverse_permute(w, nh), _adapter_reverse_permute(w, nh))
        # round-trip identity: forward(reverse(w)) == w
        assert torch.equal(_adapter_permute(tool.reverse_permute(w, nh), nh), w)
        # actually changes the layout (not a no-op)
        assert not torch.equal(tool.reverse_permute(w, nh), w)


_REAL_CKPT = "/home/alisol/projects/muse-glimmer-test/hf_ckpt"


def test_real_checkpoint_qk_if_available():
    tool = _load_tool()
    shard = os.path.join(_REAL_CKPT, "model-00001-of-00002.safetensors")
    if not os.path.exists(shard):
        return  # skip when the released checkpoint isn't on disk
    from safetensors import safe_open

    with safe_open(shard, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for suffix, nh in [("q_proj", _N_HEADS), ("k_proj", _N_KV)]:
            key = f"model.language_model.layers.0.self_attn.{suffix}.weight"
            if key not in keys:
                continue
            w = f.get_tensor(key).float()
            assert torch.equal(
                tool.reverse_permute(w, nh), _adapter_reverse_permute(w, nh)
            )
            assert torch.equal(_adapter_permute(tool.reverse_permute(w, nh), nh), w)


if __name__ == "__main__":
    test_bit_exact_vs_adapter_and_roundtrip()
    test_real_checkpoint_qk_if_available()
    print("test_convert_qk_layout: ALL PASS")
