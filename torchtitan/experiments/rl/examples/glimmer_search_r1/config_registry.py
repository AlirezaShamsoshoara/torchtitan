# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry point for the Muse Glimmer Search-R1 RL example.

Runs the multi-turn retrieval-QA Search-R1 recipe on **Muse Glimmer 30B**. The
env, dataset, and exact-match rubric are reused verbatim from the ``search_r1``
example (the recipe *is* Search-R1); only the model, renderer, parallelism, and
activation checkpointing differ. ``ConfigManager`` discovers this via::

    python -m torchtitan.experiments.rl.train \
        --module glimmer_search_r1 \
        --config rl_muse_glimmer_search_r1

Prerequisites (see ``README.md`` in this directory):
* A **Q/K-converted** Muse Glimmer checkpoint produced offline by
  ``tools/convert_qk_layout.py`` (the released HF export is in HF split-half RoPE
  layout; loading it unconverted silently scrambles attention -> reward 0). Point
  ``hf_assets_path`` at the converted dir and export
  ``GLIMMER_QK_ALREADY_CONVERTED=1`` so the state-dict adapter loads q/k verbatim.
* A retrieval server on ``message_env.search_url`` (use the real dense retriever,
  or ``standin_retriever.py`` for a plumbing-only POC).

Hard constraints baked in here (all verified against the released 30B):
* **TP <= 2 everywhere** — Muse Glimmer has 2 KV heads, so ``tensor_parallel_degree``
  must divide 2 on BOTH the vLLM generator and the trainer. This config uses generator
  TP=2 and trainer TP=2 x FSDP=3 (6 trainer GPUs + 2 generator GPUs on an 8-GPU node).
* **FullAC** — Adam m/v allocate on the first optimizer step; the default SelectiveAC
  (or too few FSDP shards, e.g. FSDP=2 on 4 GPUs) OOMs at ~92 GB/GPU.
* **varlen (FA3) attention** on both trainer and generator (H100/H200+).
"""

from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.search_r1.rollouter import SearchR1Rollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.models.muse_glimmer import model_registry

# Register the Muse Glimmer renderer ("muse_glimmer") into the renderers library
# registry so RendererConfig(name="muse_glimmer") resolves. Import for the
# side-effect; keep the reference so linters don't drop it.
from torchtitan.experiments.rl.examples.glimmer_search_r1 import renderer as _glimmer_renderer

_glimmer_renderer.register()


def rl_muse_glimmer_search_r1() -> Controller.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Muse Glimmer 30B.

    8-GPU layout: generator TP=2 (KV-head cap) + trainer FSDP/TP on the rest,
    with a retrieval server on spare capacity. FullAC on the trainer. Loads the
    (Q/K-converted) HF checkpoint on the first run, then resumes from DCP.
    """
    return Controller.Config(
        model_spec=model_registry("30B", attn_backend="varlen"),
        # Point this at the OFFLINE Q/K-CONVERTED checkpoint dir (see module
        # docstring); also export GLIMMER_QK_ALREADY_CONVERTED=1.
        hf_assets_path="./assets/hf/Muse-Glimmer-30B-qk-converted",
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=500),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=4096),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SearchR1Rollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
        # Muse Glimmer harmony + ATEM renderer (registered above). Thinking is
        # left on the renderer's own default; flip enable_thinking here if needed.
        renderer=RendererConfig(name="muse_glimmer"),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                # 30B dense: Muse Glimmer has 2 KV heads, so TP must divide 2
                # (TP<=2) on the TRAINER too, not just the generator. Shard the
                # rest of the memory with FSDP. On an 8-GPU node: 2 generator +
                # TP=2 x FSDP=3 = 6 trainer GPUs. FSDP=2 (4 GPUs) OOMs at the
                # first optimizer.step() once Adam m/v allocate under FullAC.
                data_parallel_shard_degree=3,
                tensor_parallel_degree=2,
            ),
            # FullAC is REQUIRED at this size (SelectiveAC OOMs on step 2).
            ac_config=FullAC.Config(),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,  # first run loads HF; restarts resume from DCP
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                # HARD CAP: Muse Glimmer has 2 KV heads -> generator TP <= 2.
                tensor_parallel_degree=2,
            ),
            # 0.6 (vs the 0.9 default) reserves headroom for the weight-sync
            # memory spike (same reason as the Qwen3-8B search_r1 config).
            gpu_memory_limit=0.6,
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
            ),
        ),
    )


def rl_muse_glimmer_debug_varlen_batch_invariant() -> Controller.Config:
    """Tiny Muse Glimmer debugmodel in deterministic + batch-invariant mode.

    This is the config the bitwise trainer==vLLM parity test drives (subclass of
    the upstream ``test_bitwise_parity`` harness). It exercises the Muse Glimmer
    model + the TorchTitan->vLLM generator path at ``debugmodel`` scale (random
    init, no checkpoint), so it runs on 2 GPUs in seconds and needs no weights.

    Uses the debug tokenizer (vocab 2048) and the ``qwen3`` renderer — like the
    other debug parity configs — because the debug tokenizer lacks Muse Glimmer's
    harmony/ATEM special tokens, and parity only needs trainer/generator logprob
    identity, not the real chat format. Trainer keeps fp32 master weights; FSDP
    mixed precision casts to bf16 for the forward to match the bf16 generator. TP
    must match on both sides (parity is TP-order sensitive).
    """
    from torchtitan.config import DebugConfig
    from torchtitan.experiments.rl.components.training_sample_builder import (
        TrainingSampleBuilder,
    )
    from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter
    from torchtitan.experiments.rl.losses import GRPOLoss

    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    num_samples_per_prompt = 8
    return Controller.Config(
        model_spec=model_registry("debugmodel", attn_backend="varlen"),
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=3,
            target_offpolicy_steps=0,
            window_fraction=None,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear"
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            debug=batch_invariant_config,
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                # Must match the trainer TP for bitwise parity.
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(temperature=0.8, top_p=0.95, max_tokens=50),
            debug=batch_invariant_config,
        ),
    )


def rl_muse_glimmer_search_r1_smoke() -> Controller.Config:
    """Fast, self-contained smoke variant of ``rl_muse_glimmer_search_r1``.

    Differences from the full recipe (everything else inherited):
    * Tiny run: 2 training steps, few prompts/samples, short completions.
    * Local QA parquet fixture (``assets/qa_smoke.parquet``) + the stand-in
      retriever (oracle mode) so it runs offline with no dense index and produces
      a non-degenerate reward signal to exercise the loop.
    * ``cudagraph.enable=False`` -> ``enforce_eager`` on the generator, to bypass a
      known torch-nightly ``torch.compile`` ``aot_compile`` pickling bug
      (``WeakValueDictionary``) unrelated to Muse Glimmer.

    Intended to prove the full RL loop end-to-end (weight-sync, no OOM under
    FullAC + TP=2, reward flows) on the real converted 30B checkpoint. NOT a
    convergence run.
    """
    import dataclasses
    import os

    from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Dataset

    config = rl_muse_glimmer_search_r1()

    # Fresh dump folder so the trainer does NOT resume from a stale DCP
    # checkpoint left by an earlier run (the default outputs/rl/checkpoint may
    # contain incompatible steps -> "Missing key ... o_gate" on DCP load). A
    # clean folder forces the intended HF initial-load path.
    config.dump_folder = "outputs/rl_glimmer_smoke"

    # Short run.
    config.async_loop.num_training_steps = 2
    config.async_loop.num_prompts_per_train_step = 2
    config.async_loop.num_samples_per_prompt = 4
    config.async_loop.validation = ValidationConfig(num_samples=0)
    config.async_loop.batcher = Batcher.Config(
        batch=BatchConfig(local_batch_size=1, seq_len=2048)
    )

    # Local QA fixture for both train and (skipped) validation.
    _here = os.path.dirname(__file__)
    qa_path = os.path.join(_here, "assets", "qa_smoke.parquet")
    config.rollouter.train_dataset = SearchR1Dataset.Config(
        data_path=qa_path, seed=42
    )
    config.rollouter.validation_dataset = SearchR1Dataset.Config(
        data_path=qa_path, seed=99, shuffle=False
    )

    # Shorter generations for speed.
    config.generator = dataclasses.replace(
        config.generator,
        cudagraph=VLLMCudagraphConfig(enable=False),
        sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=256),
    )
    return config


def rl_muse_glimmer_search_r1_100step() -> Controller.Config:
    """100-step offline RL run for Muse Glimmer 30B on a no-InfiniBand box.

    Same recipe as ``rl_muse_glimmer_search_r1`` but: local QA parquet + stand-in
    retriever (offline), validation skipped (num_samples=0), short completions for
    throughput, and 100 training steps so we can see the reward/loss trajectory.
    Requires ``USE_TORCHCOMMS_RDMA=0`` on the launcher (disables the torchcomms
    RDMA/ibverbs weight-sync transport that hangs without InfiniBand).
    """
    import dataclasses
    import os

    from torchtitan.experiments.rl.components.training_sample_builder import (
        TrainingSampleBuilder,
    )
    from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Dataset
    from torchtitan.experiments.rl.examples.search_r1.rubric import RewardExactMatch
    from torchtitan.experiments.rl.rubrics.rubric import Rubric

    config = rl_muse_glimmer_search_r1()
    config.dump_folder = "outputs/rl_glimmer_100step"
    # Memory-safe single-node layout for a long run on 8xH100 (no IB).
    # The default TP=2 x FSDP=3 (6 trainer GPUs) leaves the 30B weights + Adam
    # m/v + FullAC too tightly packed and OOMs device 0 within a few steps.
    # Shard the trainer across 7 GPUs with pure FSDP (TP=1 is valid — TP only
    # has to divide the 2 KV heads) and drop the generator to TP=1 (1 GPU), so
    # optimizer/param/grad memory is sharded 7-way instead of 3-way.
    config.trainer.parallelism = ParallelismConfig(
        data_parallel_shard_degree=7,
        tensor_parallel_degree=1,
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=InferenceParallelismConfig(
            data_parallel_degree=1, tensor_parallel_degree=1
        ),
    )
    config.async_loop.num_training_steps = 100
    # The periodic DCP checkpoint save is ~130 GB across 7 ranks; on this shared
    # box that I/O storm spikes host load enough to knock the run over (observed:
    # runs died exactly at the interval=50 save). Push the interval past the run
    # length so no mid-run save happens, but keep checkpointing enabled so the
    # initial converted-HF load path (initial_load_in_hf) still works.
    config.trainer.checkpoint = CheckpointManager.Config(
        enable=True,
        initial_load_in_hf=True,
        interval=1000,
        last_save_model_only=True,
        keep_latest_k=2,
    )
    config.async_loop.num_prompts_per_train_step = 4
    config.async_loop.num_samples_per_prompt = 6
    # Bound replay-buffer memory: active_slots = (target_offpolicy_steps+1) *
    # num_prompts_per_train_step. The default target=3 -> 16 in-flight groups,
    # which OOM'd the trainer (device 0) around step 9 on this 8xH100 box.
    # target=1 -> 8 groups keeps the async pipeline but roughly halves the
    # generator KV + buffered-activation high-water mark.
    config.async_loop.target_offpolicy_steps = 1
    config.async_loop.validation = ValidationConfig(num_samples=0)  # skip (offline)
    config.async_loop.batcher = Batcher.Config(
        batch=BatchConfig(local_batch_size=1, seq_len=1536)
    )
    # Keep zero-variance groups so the loop still steps + logs a reward/loss
    # trajectory even before the reward signal separates (belt-and-suspenders;
    # the reward levers below are what actually create the variance).
    config.async_loop.training_sample_builder = TrainingSampleBuilder.Config(
        drop_zero_std_reward_groups=False,
    )
    qa = os.path.join(os.path.dirname(__file__), "assets", "qa_smoke.parquet")
    config.rollouter.train_dataset = SearchR1Dataset.Config(data_path=qa, seed=42)
    config.rollouter.validation_dataset = SearchR1Dataset.Config(
        data_path=qa, seed=99, shuffle=False
    )
    # Put search on the gradient so rewards are non-degenerate on the offline
    # stand-in oracle: a correct answer that skipped search is penalised, and a
    # wrong/empty answer still earns partial credit if a search surfaced the
    # gold span. Combined with the fuzzy-matching oracle retriever this yields
    # real reward variance -> trainable groups -> a moving reward/loss curve.
    config.rollouter.rubric = Rubric.Config(
        reward_fns=[
            RewardExactMatch.Config(
                weight=1.0, no_search_penalty=0.2, retrieval_score=0.3
            ),
        ],
        truncation_reward=0.0,
    )
    config.generator = dataclasses.replace(
        config.generator,
        cudagraph=VLLMCudagraphConfig(enable=False),
        sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=384),
    )
    return config
