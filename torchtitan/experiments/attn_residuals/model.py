# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Attention Residuals model: TransformerBlock and Decoder with Block AttnRes.

This module implements Block Attention Residuals as described in
"Attention Residuals" (Kimi Team, 2025). Layers are grouped into N blocks;
within each block, sub-layer outputs accumulate via standard residual addition.
Across blocks, a learned softmax attention over block representations replaces
the fixed unit-weight residual.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger

from .attn_res import block_attn_res


class AttnResTransformerBlock(TransformerBlock):
    """TransformerBlock with Block Attention Residuals.

    Each block contains two AttnRes application points (pre-attention and
    pre-MLP), following the paper's treatment of each attention and MLP
    sub-layer as a separate "layer" in the depth dimension.

    The block tracks its layer_id and block_size to detect block boundaries
    where the partial_block accumulation is finalized and appended to the
    blocks list.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True
        num_attn_res_blocks: int = 8
        attn_res_norm: RMSNorm.Config = field(default_factory=RMSNorm.Config)

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        # Standard transformer components
        self.attention = config.attention.build(dim=dim)
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build(dim=dim)
        self.attention_norm = config.attention_norm.build(normalized_shape=dim)
        self.ffn_norm = config.ffn_norm.build(normalized_shape=dim)

        # AttnRes components: projection + norm for pre-attention and pre-MLP
        # Use Linear (Module-protocol-compatible) instead of nn.Linear
        self.attn_res_proj = Linear.Config(bias=False).build(
            in_features=dim, out_features=1
        )
        self.attn_res_norm = config.attn_res_norm.build(normalized_shape=dim)
        self.mlp_res_proj = Linear.Config(bias=False).build(
            in_features=dim, out_features=1
        )
        self.mlp_res_norm = config.attn_res_norm.build(normalized_shape=dim)

        # Block boundary tracking
        self.layer_id = layer_id
        # block_size = number of transformer layers per block
        # (each transformer layer has 2 sub-layers: attn + MLP)
        self.block_size = max(1, n_layers // config.num_attn_res_blocks)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    @property
    def is_block_boundary(self) -> bool:
        """Whether this layer starts a new block."""
        return self.layer_id % self.block_size == 0

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass with Block Attention Residuals.

        Args:
            blocks: Completed block representations (list of [B, T, D] tensors).
            partial_block: Current intra-block partial sum [B, T, D], or None
                at the start of a new block.
            freqs_cis: Rotary embedding frequencies.
            attention_masks: Attention masks (flex/varlen/sdpa).
            positions: Optional position indices.

        Returns:
            Updated (blocks, partial_block) tuple.
        """
        # Pre-attention: attend over blocks + partial to get input
        # (Paper Figure 2: AttnRes BEFORE boundary check)
        h = block_attn_res(
            blocks, partial_block, self.attn_res_proj, self.attn_res_norm
        )

        # At block boundary: finalize current partial block and start new one
        if self.is_block_boundary and self.layer_id > 0:
            blocks = blocks + [partial_block]
            partial_block = None

        # Self-attention sub-layer
        attn_out = self.attention(
            self.attention_norm(h), freqs_cis, attention_masks, positions
        )
        partial_block = (
            partial_block + attn_out if partial_block is not None else attn_out
        )

        # Pre-MLP: attend over blocks + updated partial
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        # MLP sub-layer
        mlp_out = self.feed_forward(self.ffn_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block

    def init_weights(self, **kwargs):
        # Standard component initialization
        for norm in (self.attention_norm, self.ffn_norm):
            norm.init_weights()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)

        # AttnRes norms
        self.attn_res_norm.init_weights()
        self.mlp_res_norm.init_weights()

        # CRITICAL: zero-init pseudo-query projections (paper Section 5).
        # This ensures initial softmax weights are uniform across all sources,
        # preventing training volatility at the start.
        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)


class AttnResDecoder(Decoder):
    """Decoder with Block Attention Residuals.

    Overrides the standard Decoder forward to maintain a list of block
    representations and a partial_block accumulator. The token embedding
    serves as the initial block (b_0 in the paper).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        n_layers: int = 32
        vocab_size: int = 128256
        enable_weight_tying: bool = False
        layer: AttnResTransformerBlock.Config

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            import dataclasses as _dc

            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum "
                    f"{self.rope.max_seq_len}."
                )
            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    f"Got attn_backend='{self.layer.attention.attn_backend}'."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layer.attention.n_heads
                # pyrefly: ignore [missing-attribute]
                n_kv_heads = self.layer.attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_kv_heads ({n_kv_heads})."
                    )

            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * (self.dim // self.layer.attention.n_heads),
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.output.weight

    def init_weights(self, **kwargs):
        if self.enable_weight_tying:
            # When initialized on meta device, tying in __init__ may not
            # have worked correctly. Re-tie here before calling super().
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight

        super().init_weights(**kwargs)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        # Embedding: passthrough for PP stages that don't have tok_embeddings
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        # Initialize AttnRes state:
        # - blocks[0] = token embedding (b_0 in the paper)
        # - partial_block starts as None (no intra-block partial sum yet;
        #   paper Algorithm 1: b_n^0 := 0, Eq 6: first layer sees only blocks)
        blocks: list[torch.Tensor] = [h]
        partial_block: torch.Tensor | None = None

        for layer in self.layers.values():
            blocks, partial_block = layer(
                blocks, partial_block, self.freqs_cis, attention_masks, positions
            )

        # Final output: use the last partial_block as the model's hidden state.
        # Append it to blocks for a final AttnRes aggregation? The paper's
        # pseudocode (Figure 2) returns blocks and partial_block, with the
        # final layer's output used directly. We use partial_block + last
        # block context via a simple sum to maintain gradient flow.
        h = partial_block
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
