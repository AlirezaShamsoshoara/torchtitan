#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fast generation sanity check for a Muse Glimmer RL checkpoint.

This is the "is the checkpoint sane?" gate to run BEFORE a full RL run. It loads
the checkpoint through the same TorchTitan->vLLM registration the RL generator
uses, renders a Search-R1-style prompt with the Muse Glimmer renderer, generates
a short completion, and reports the signals that distinguish a correctly-loaded
checkpoint from a mis-loaded one:

  * ``finish_reason`` -> expect ``stop`` (model chose to stop), NOT ``length``
    (ran to max_tokens producing garbage — the classic symptom of the Q/K RoPE
    layout NOT being converted).
  * coherent text (eyeball).
  * a parseable ATEM tool call when a ``search`` tool is offered.

If you see ``finish_reason=length`` + gibberish + no tool call, the checkpoint
was almost certainly loaded WITHOUT the offline Q/K conversion (run
``tools/convert_qk_layout.py`` and set ``GLIMMER_QK_ALREADY_CONVERTED=1``).

Run (single GPU is fine for TP=1; use torchrun --nproc_per_node=2 for TP=2):

    GLIMMER_QK_ALREADY_CONVERTED=1 \
    torchrun --nproc_per_node=2 \
      -m torchtitan.experiments.rl.examples.glimmer_search_r1.tools.smoke_generate \
      --checkpoint <converted_ckpt_dir> \
      --prompt "Who wrote the novel Frankenstein?"

By default it uses the ``rl_muse_glimmer_search_r1`` config's generator settings;
``--checkpoint`` overrides ``hf_assets_path`` so you can point at the converted
checkpoint without editing the config.
"""

from __future__ import annotations

import argparse
import os

# Must precede any CUDA / vLLM import.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.experiments.rl.examples.glimmer_search_r1 import config_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
)
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.tools.utils import has_cuda_capability

logger = init_logger(__name__)

# A minimal Search-R1-style search tool so the model can emit an ATEM tool call.
_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search a knowledge base and return the top passages for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Muse Glimmer RL checkpoint smoke test.")
    ap.add_argument("--config", default="rl_muse_glimmer_search_r1")
    ap.add_argument("--checkpoint", default=None, help="override hf_assets_path (converted ckpt dir)")
    ap.add_argument("--prompt", default="Who wrote the novel Frankenstein? Search if unsure.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-tool", action="store_true", help="don't offer the search tool")
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument(
        "--eager",
        action="store_true",
        help="force enforce_eager (bypass vLLM torch.compile/cudagraph). Useful to "
        "isolate generation from nightly torch.compile issues.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("GLIMMER_QK_ALREADY_CONVERTED"):
        logger.warning(
            "GLIMMER_QK_ALREADY_CONVERTED is not set. If --checkpoint is a "
            "Q/K-converted checkpoint you MUST set it to 1, or the adapter will "
            "double-permute and produce gibberish."
        )

    config_factory = getattr(config_registry, args.config, None)
    if not callable(config_factory):
        raise ValueError(f"Unknown RL config {args.config!r}")
    config = config_factory()
    gen_config = config.generator
    model_spec = config.model_spec
    model_path = args.checkpoint or config.hf_assets_path
    is_rank0 = os.environ.get("RANK", "0") == "0"

    register_to_vllm(
        model_spec,
        parallelism=gen_config.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=model_path,
        ),
        override=gen_config.override,
    )

    inner_attn = model_spec.model.layers[0].attention.inner_attention
    if not isinstance(inner_attn, (VarlenAttention.Config, FlexAttention.Config)):
        raise ValueError("Only varlen and flex attention backends are supported.")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        data_parallel_size=gen_config.parallelism.data_parallel_degree,
        worker_cls=TORCHTITAN_WORKER_CLS,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=args.eager or (not gen_config.cudagraph.enable),
        attention_config=AttentionConfig(
            backend=(
                AttentionBackendEnum.FLEX_ATTENTION
                if isinstance(inner_attn, FlexAttention.Config)
                else AttentionBackendEnum.CUSTOM
            ),
        ),
        max_model_len=model_spec.model.max_seq_len,
        max_num_seqs=args.max_num_seqs,
    )
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256
    engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))

    renderer = config.renderer.build(tokenizer_path=model_path)
    stop_token_ids = list(renderer.get_stop_token_ids())
    tools = None if args.no_tool else [_SEARCH_TOOL]

    prompt_token_ids = renderer.render_ids(
        messages=[{"role": "user", "content": args.prompt}],
        tools=tools,
        add_generation_prompt=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        n=1,
        stop_token_ids=stop_token_ids or None,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    engine_input = engine.renderer.render_cmpl([{"prompt_token_ids": prompt_token_ids}])[0]
    engine.add_request("0", engine_input, sampling_params)

    while engine.has_unfinished_requests():
        for out in engine.step():
            if not out.finished:
                continue
            comp = out.outputs[0]
            finish_reason = comp.finish_reason
            text = comp.text
            n_out = len(comp.token_ids)
            parsed = renderer.parse_response(list(comp.token_ids), tools=tools)
            ok_calls = [
                tc for tc in parsed.tool_calls
                if getattr(tc, "status", None) is not None
                and str(tc.status).endswith("ok")
            ]
            if is_rank0:
                print("\n===== Muse Glimmer smoke_generate =====", flush=True)
                print(f"prompt: {args.prompt}", flush=True)
                print(f"finish_reason: {finish_reason}  (want 'stop', NOT 'length')", flush=True)
                print(f"generated_token_count: {n_out}", flush=True)
                print(f"parsed tool calls (OK): {[(tc.name, tc.arguments) for tc in ok_calls]}", flush=True)
                print(f"text: {text!r}", flush=True)
                verdict = "PASS" if finish_reason == "stop" else "SUSPECT (length -> check Q/K conversion)"
                print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
