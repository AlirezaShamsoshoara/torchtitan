# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline Q/K RoPE-layout conversion for Muse Glimmer RL.

Why this tool exists (the non-obvious part)
--------------------------------------------
The released ``meta-models/Muse-Glimmer-30B`` HF export stores ``q_proj`` /
``k_proj`` rows in HF's **split-half** ``rotate_half`` RoPE layout
(``x1 = x[..., : d/2]``, ``x2 = x[..., d/2 :]``). torchtitan's ``ComplexRoPE``
pairs **adjacent** head-dim components (``view_as_complex(reshape(..., -1, 2))``).
The two conventions are related by a per-head row permutation of the q/k
projection weights (the same permute ``Llama3StateDictAdapter`` applies).

``MuseGlimmerStateDictAdapter.from_hf`` already applies that permute — and that
is correct for the trainer's *own* checkpoint save/load. **But for RL it is not
enough**, and doing the permute *only* in the adapter is actively dangerous:

* Under the RL trainer's ``FSDP x TP`` sharding, q/k weights are
  ``_StridedShard`` DTensors sharded on **dim 0** — exactly the dim the permute
  reshapes. The head-splitting ``view(n_heads, 2, ...)`` cannot unflatten an
  unevenly/strided-sharded dim; ``redistribute -> Replicate -> permute ->
  redistribute`` cannot faithfully rebuild a *strided* shard after a gather.
* The result: an adapter-side permute silently converts only the plain-tensor
  (generator) path and leaves the **trainer** on the wrong layout. There is **no
  error** — training simply produces scrambled attention: gibberish rollouts,
  every completion truncated, reward ~0, entropy far above healthy (~0.2).

So for the RL path we convert the checkpoint **once, offline**, on plain CPU
tensors (no DTensor, no sharding), and then run training with the adapter's
permute **disabled** (``GLIMMER_QK_ALREADY_CONVERTED=1`` / the RL config sets
``qk_layout_preconverted=True``). This keeps the trainer and generator on the
same, correct layout.

What it does
------------
Reads a HF ``meta-models/Muse-Glimmer-30B`` checkpoint (safetensors shards),
applies the split-half -> interleaved reverse-permute to **only** the text
decoder ``q_proj`` / ``k_proj`` weight rows (identical math to
``MuseGlimmerStateDictAdapter._reverse_permute``), and writes a new checkpoint
that torchtitan can load directly for RL. Every other tensor is copied verbatim.

Usage
-----
    python -m torchtitan.experiments.rl.examples.glimmer_search_r1.tools.convert_qk_layout \
        <hf_export_dir> <converted_out_dir> \
        [--n-heads 32 --n-kv-heads 2 --head-dim 128 --dim 6656]

The head geometry defaults to the released 30B config and is otherwise read
from the export's ``config.json`` when present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch


# ---- permute math: identical to MuseGlimmerStateDictAdapter._reverse_permute ----
# HF split-half row order -> torchtitan ComplexRoPE interleaved row order.
def reverse_permute(w: torch.Tensor, n_heads_arg: int) -> torch.Tensor:
    """split-half (HF) -> interleaved (titan) for a q/k projection weight.

    ``w`` has shape ``[n_heads_arg * head_dim, in_dim]`` (rows are the output
    features grouped by head). Mirrors the adapter exactly.
    """
    dim1, dim2 = w.shape
    return (
        w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
        .clone()
    )


def _load_index(hf_dir: str) -> tuple[dict[str, str], list[str]]:
    """Return (weight_map, shard_files). Supports sharded or single-file."""
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        return weight_map, shards
    # single-file fallback
    single = "model.safetensors"
    if os.path.exists(os.path.join(hf_dir, single)):
        return {}, [single]
    raise FileNotFoundError(
        f"No model.safetensors.index.json or model.safetensors in {hf_dir}"
    )


def _read_geometry(hf_dir: str, args: argparse.Namespace) -> tuple[int, int]:
    """Return (n_heads, n_kv_heads) from CLI overrides or config.json."""
    n_heads, n_kv_heads = args.n_heads, args.n_kv_heads
    cfg_path = os.path.join(hf_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        # released config nests text config under text_config for the VLM
        text = cfg.get("text_config", cfg)
        n_heads = n_heads or text.get("num_attention_heads")
        n_kv_heads = n_kv_heads or text.get("num_key_value_heads")
    if not n_heads or not n_kv_heads:
        raise ValueError(
            "Could not determine head geometry; pass --n-heads and --n-kv-heads."
        )
    return int(n_heads), int(n_kv_heads)


# text-decoder q/k weight keys in the released HF export
_Q_SUFFIX = "self_attn.q_proj.weight"
_K_SUFFIX = "self_attn.k_proj.weight"
_TEXT_PREFIX = "model.language_model."


def _is_text_q(key: str) -> bool:
    return key.startswith(_TEXT_PREFIX) and key.endswith(_Q_SUFFIX)


def _is_text_k(key: str) -> bool:
    return key.startswith(_TEXT_PREFIX) and key.endswith(_K_SUFFIX)


def convert(hf_dir: str, out_dir: str, args: argparse.Namespace) -> dict[str, int]:
    """Convert q/k layout for every text-decoder shard; copy the rest verbatim.

    Returns a small stats dict for logging/verification.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    n_heads, n_kv_heads = _read_geometry(hf_dir, args)
    weight_map, shards = _load_index(hf_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats = {"q_converted": 0, "k_converted": 0, "copied": 0, "shards": len(shards)}

    for shard in shards:
        src = os.path.join(hf_dir, shard)
        dst = os.path.join(out_dir, shard)
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(src, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if _is_text_q(key):
                    t = reverse_permute(t, n_heads)
                    stats["q_converted"] += 1
                elif _is_text_k(key):
                    t = reverse_permute(t, n_kv_heads)
                    stats["k_converted"] += 1
                else:
                    stats["copied"] += 1
                out_tensors[key] = t
        save_file(out_tensors, dst, metadata={"format": "pt"})
        print(f"[convert_qk_layout] wrote {dst} ({len(out_tensors)} tensors)")

    # Copy non-weight assets (config, tokenizer, index) verbatim so the output
    # dir is a drop-in checkpoint.
    for name in os.listdir(hf_dir):
        if name.endswith(".safetensors"):
            continue
        s = os.path.join(hf_dir, name)
        d = os.path.join(out_dir, name)
        if os.path.isfile(s):
            shutil.copy2(s, d)

    print(
        f"[convert_qk_layout] done: q={stats['q_converted']} k={stats['k_converted']} "
        f"copied={stats['copied']} across {stats['shards']} shard(s) "
        f"(n_heads={n_heads}, n_kv_heads={n_kv_heads})"
    )
    print(
        "[convert_qk_layout] NOTE: run RL with the adapter permute DISABLED "
        "(qk_layout_preconverted=True / GLIMMER_QK_ALREADY_CONVERTED=1)."
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Offline Q/K RoPE layout conversion for Muse Glimmer RL.")
    ap.add_argument("hf_dir", help="input HF checkpoint dir (meta-models/Muse-Glimmer-30B export)")
    ap.add_argument("out_dir", help="output dir for the converted checkpoint")
    ap.add_argument("--n-heads", type=int, default=None, help="override num_attention_heads")
    ap.add_argument("--n-kv-heads", type=int, default=None, help="override num_key_value_heads")
    ap.add_argument("--head-dim", type=int, default=None, help="(informational) head_dim")
    ap.add_argument("--dim", type=int, default=None, help="(informational) model dim")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    convert(args.hf_dir, args.out_dir, args)


if __name__ == "__main__":
    main()
