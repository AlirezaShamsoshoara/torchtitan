# Attention Residuals for Qwen3 (MoE)

## Overview

This experiment implements **Block Attention Residuals (AttnRes)** for the
**Qwen3** model family, including its Mixture-of-Experts (MoE) variants. The
original AttnRes experiment (`attn_residuals/`) targets the dense Llama3
architecture. This experiment extends AttnRes to Qwen3 to evaluate its
effectiveness on MoE models --- which is notably the same architecture class
used in the original paper (Kimi Linear, a 48B/3B MoE).

## Motivation

The Attention Residuals paper (Kimi Team, 2025) conducted all experiments on
MoE models, yet our initial implementation only covered dense Llama3. Key
questions this experiment aims to answer:

1. **Does AttnRes improve loss on MoE models in our framework?** The paper
   reports 1.25x compute advantage on MoE; we need to verify this holds in
   torchtitan's Qwen3 implementation.
2. **What is the TPS overhead with MoE?** MoE already has routing/permutation
   overhead. AttnRes adds depth-attention overhead on top. We need to quantify
   the combined cost.
3. **How does AttnRes interact with Expert Parallelism (EP)?** The depth
   attention operates on full hidden states before/after MoE layers. EP
   changes token distribution within MoE but should not affect AttnRes.
4. **Dense vs MoE comparison**: Qwen3 has both dense (0.6B-32B) and MoE
   (30B-A3B, 235B-A22B) variants, enabling direct comparison of AttnRes
   effectiveness across architectures.

## Architecture

AttnRes replaces fixed residual connections with learned softmax attention over
depth. Layers are grouped into N blocks (~8). Within blocks, standard residual
addition is used. Across blocks, a learned pseudo-query vector computes
softmax-weighted depth attention over all previous block representations.

### Key modifications over base Qwen3

Each transformer layer gains 4 extra parameters (negligible overhead):
- `attn_res_proj` (Linear d->1): pre-attention pseudo-query
- `attn_res_norm` (RMSNorm): pre-attention key normalization
- `mlp_res_proj` (Linear d->1): pre-MLP/MoE pseudo-query
- `mlp_res_norm` (RMSNorm): pre-MLP/MoE key normalization

The forward pass threads a `(blocks, partial_block)` state through layers
instead of a single hidden state tensor.

### MoE interaction

AttnRes is **agnostic to the feed-forward type**. The depth attention operates
on hidden states at the layer level. Whether the FFN sub-layer is dense
(`FeedForward`) or MoE (`MoE` with routing), AttnRes treats its output
identically. This matches the paper's design where MoE layers are used
throughout.

## Qwen3-specific considerations

- **QK normalization**: Qwen3 applies RMSNorm to Q/K before RoPE. This is
  orthogonal to AttnRes (which normalizes depth-wise keys, not attention keys).
- **RoPE cos_sin backend**: Qwen3 uses `cos_sin` RoPE instead of Llama3's
  `complex`. No impact on AttnRes (RoPE is internal to attention).
- **Expert Parallelism (EP)**: AttnRes operates on the full hidden state
  before/after MoE. EP only affects token distribution within the MoE layer.
  No conflict expected.
- **Pipeline Parallelism (PP)**: Requires block caching (paper Section 4.1)
  because the growing `blocks` list is incompatible with PP's fixed-size
  inter-stage tensor communication. Each stage receives/sends a single tensor,
  maintaining blocks locally. Implemented in Phase 6.
- **Weight tying**: Qwen3 dense models up to 4B use weight tying.
  AttnRes supports this (validated in Llama3 experiment).

## File structure

```
attn_residuals_qwen/
  __init__.py          -- Model registration and config definitions
  model.py             -- AttnResQwen3TransformerBlock, AttnResQwen3Model
  parallelize.py       -- Parallelism: TP, EP, FSDP, AC, compile
  config_registry.py   -- Trainer configs for experiments
  tests/
    __init__.py
    test_attn_res.py   -- Core block_attn_res tests (MoE context)
    test_model.py      -- Block and decoder unit tests
    test_parallelize.py -- TP plan, fake backend, EP tests
```

## Running

```bash
# Unit tests
python -m pytest torchtitan/experiments/attn_residuals_qwen/tests/ -x -v

# Debug model training (dense, fake backend)
NGPU=4 LOCAL_RANK=0 python -m torchtitan.train \
  --module attn_residuals_qwen --model.flavor debugmodel \
  --comm.mode=fake_backend

# Debug model training (MoE, fake backend)
NGPU=4 LOCAL_RANK=0 python -m torchtitan.train \
  --module attn_residuals_qwen --model.flavor debugmodel_moe \
  --comm.mode=fake_backend

# Multi-GPU training
torchrun --nproc_per_node=8 -m torchtitan.train \
  --module attn_residuals_qwen --model.flavor debugmodel_moe \
  --job.config.file attn_res_qwen3_moe_debug
```

## References

- Paper: "Attention Residuals" (Kimi Team / Moonshot AI, 2025)
- Original Llama3 implementation: `torchtitan/experiments/attn_residuals/`
- Qwen3 base model: `torchtitan/models/qwen3/`
