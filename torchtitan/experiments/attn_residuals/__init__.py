# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    FeedForward,
    GQAttention,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.protocols.model_spec import ModelSpec

from .model import AttnResDecoder, AttnResTransformerBlock
from .parallelize import parallelize_attn_res

__all__ = [
    "parallelize_attn_res",
    "AttnResDecoder",
    "AttnResTransformerBlock",
    "attn_res_configs",
]


attn_res_configs = {
    "debugmodel": AttnResDecoder.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=AttnResTransformerBlock.Config(
            num_attn_res_blocks=3,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            attn_res_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "1B": AttnResDecoder.Config(
        dim=2048,
        n_layers=16,
        vocab_size=128256,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=AttnResTransformerBlock.Config(
            num_attn_res_blocks=8,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            attn_res_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    2048, multiple_of=1024, ffn_dim_multiplier=1.5
                ),
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=2048 // 32,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "debugmodel_v2": AttnResDecoder.Config(
        dim=256,
        n_layers=32,
        vocab_size=128256,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=AttnResTransformerBlock.Config(
            num_attn_res_blocks=8,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            attn_res_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "8B": AttnResDecoder.Config(
        dim=4096,
        n_layers=32,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=AttnResTransformerBlock.Config(
            num_attn_res_blocks=8,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            attn_res_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    4096, multiple_of=1024, ffn_dim_multiplier=1.3
                ),
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="attn_residuals",
        flavor=flavor,
        model=attn_res_configs[flavor],
        parallelize_fn=parallelize_attn_res,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
