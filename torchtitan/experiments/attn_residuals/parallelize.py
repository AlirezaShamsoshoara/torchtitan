# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization for AttnRes models.

Extends the standard Llama3 parallelization with TP plans for the
AttnRes-specific modules (attn_res_proj, mlp_res_proj, attn_res_norm,
mlp_res_norm).
"""

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile_dense
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger

from .model import AttnResDecoder


def parallelize_attn_res(
    model: AttnResDecoder,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply TP, AC, compile, and FSDP to the AttnRes model.

    The TP plan extends the standard Llama3 plan with entries for the
    AttnRes norms (SequenceParallel). The proj weights are [1, D] and
    operate in the replicated domain — no TP plan needed for them.
    """
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"Sequence length {training.seq_len} must be divisible by the product "
        f"of TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        _apply_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    attn_backend = model.config.layer.attention.attn_backend
    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
            attn_backend,
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile_dense(model, compile_config)

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        _apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )
        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the AttnRes model")
        else:
            logger.info("Applied FSDP to the AttnRes model")
    elif parallel_dims.dp_replicate_enabled:
        _apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model


def _apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
):
    """Apply tensor parallelism with AttnRes-specific plan entries."""
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    for transformer_block in model.layers.values():
        layer_plan = {
            # Standard attention TP plan
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            # Standard FFN TP plan
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
            # AttnRes norms: SequenceParallel since they operate on
            # hidden states that are Shard(1) between blocks
            "attn_res_norm": SequenceParallel(),
            "mlp_res_norm": SequenceParallel(),
            # attn_res_proj and mlp_res_proj are handled below via
            # distribute_tensor (parallelize_module only handles submodules,
            # not parameter-level placements).
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

        # Distribute proj weights as Replicate DTensors on the TP mesh so
        # element-wise ops with Shard(1) hidden states don't trigger mixed
        # Tensor/DTensor errors. Must be done after parallelize_module since
        # it only handles submodules, not individual parameters.
        for proj in (transformer_block.attn_res_proj, transformer_block.mlp_res_proj):
            proj.weight = nn.Parameter(
                distribute_tensor(proj.weight, tp_mesh, [Replicate()]),
                requires_grad=proj.weight.requires_grad,
            )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the AttnRes model"
    )


def _disable_fsdp_gradient_division(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def _apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if getattr(model, "enable_weight_tying", False):
        # When weights are tied, tok_embeddings and output share the same
        # parameter. Group them in one FSDP unit to avoid duplicate all-gathers.
        modules = [
            m for m in (model.tok_embeddings, model.norm, model.output) if m is not None
        ]
        fully_shard(
            modules,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    else:
        if model.tok_embeddings is not None:
            fully_shard(
                model.tok_embeddings,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        if model.norm is not None and model.output is not None:
            fully_shard(
                [model.norm, model.output],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )
    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config)
    _disable_fsdp_gradient_division(model)


def _apply_replicate(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    replicate_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if model.tok_embeddings is not None:
        replicate(model.tok_embeddings, **replicate_config)
    for layer_id, transformer_block in model.layers.items():
        replicate(transformer_block, **replicate_config)
    if model.norm is not None and model.output is not None:
        replicate([model.norm, model.output], **replicate_config)
    replicate(model, **replicate_config)
    _disable_fsdp_gradient_division(model)
    logger.info("Applied replicate to the AttnRes model")
