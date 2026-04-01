# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Block Attention Residuals (AttnRes) core module.

Implements the depth-wise softmax attention mechanism from the paper
"Attention Residuals" (Kimi Team, 2025). Instead of fixed unit-weight
residual connections, each layer attends over previous block representations
using a learned pseudo-query vector.

Reference: Figure 2 and Equations 2-6 in the paper.
"""

import torch
import torch.nn as nn


def _ensure_dtensors(
    sources: list[torch.Tensor], ref_dtensor: torch.Tensor
) -> list[torch.Tensor]:
    """Convert sources to DTensors if weights are DTensors but sources aren't.

    Under TP, hidden states may be AsyncCollectiveTensors or plain Tensors
    (e.g. from torch.zeros_like on an async tensor). The weighted sum requires
    all operands to be DTensors when weights are DTensors.
    """
    try:
        from torch.distributed.tensor import DTensor, Shard
    except ImportError:
        return sources

    if not isinstance(ref_dtensor, DTensor):
        return sources
    if all(isinstance(v, DTensor) for v in sources):
        return sources

    mesh = ref_dtensor.device_mesh
    # Sources are [B, T, D] with T sharded under sequence parallel
    placements = [Shard(1)]
    result = []
    for v in sources:
        if isinstance(v, DTensor):
            result.append(v)
        else:
            # Resolve AsyncCollectiveTensor to plain tensor
            local = v.wait() if hasattr(v, "wait") else v
            result.append(DTensor.from_local(local, mesh, placements, run_check=False))
    return result


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.RMSNorm,
) -> torch.Tensor:
    """Compute inter-block attention: softmax-weighted sum over block representations.

    The attention weights are computed as:
        alpha_{i->l} = softmax_i(w_l^T @ RMSNorm(v_i))
    where w_l is the pseudo-query (proj.weight) and v_i are block representations.

    Args:
        blocks: Completed block representations, each of shape [B, T, D].
        partial_block: Current intra-block partial sum of shape [B, T, D].
        proj: Linear(d, 1, bias=False) — learned pseudo-query projection.
        norm: RMSNorm(d) — key normalization to prevent magnitude-dominant layers.

    Returns:
        Attended representation of shape [B, T, D].
    """
    # Compute per-source to avoid torch.stack view issues with DTensor sharding
    sources = blocks + [partial_block]

    # logits: [N+1, B, T] — depth-wise attention scores per source
    # Use element-wise mul + sum instead of matmul to avoid aten.view
    # flattening the sharded sequence dim under TP.
    w = proj.weight  # [1, D]
    logits = torch.stack([(norm(v) * w).sum(dim=-1) for v in sources])

    # weights: [N+1, B, T] — softmax over depth (dim=0)
    weights = logits.softmax(dim=0)

    # Under TP, sources may be non-DTensor (AsyncCollectiveTensor or plain
    # Tensor) while weights are DTensors. Convert for compatible ops.
    sources = _ensure_dtensors(sources, weights)

    # h: [B, T, D] — weighted sum over sources
    h = sum(w_i.unsqueeze(-1) * v_i for w_i, v_i in zip(weights, sources))

    return h
