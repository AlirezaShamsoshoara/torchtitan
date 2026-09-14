# Muse Glimmer — Search-R1 RL example

Runs the multi-turn retrieval-QA **Search-R1** recipe on **Muse Glimmer 30B**
(`meta-models/Muse-Glimmer-30B`) with TorchTitan RL (GRPO/DAPO). The dataset,
environment, and exact-match rubric are reused verbatim from the `search_r1`
example; this package only adds the Glimmer-specific glue:

| File | Purpose |
|---|---|
| `config_registry.py` | `rl_muse_glimmer_search_r1` (30B recipe), `rl_muse_glimmer_search_r1_smoke` (fast offline POC), and `rl_muse_glimmer_debug_varlen_batch_invariant` (bitwise-parity harness config). |
| `renderer.py` | `MuseGlimmerRenderer` — harmony chat frame + `to=self` reasoning + ATEM tool calls; registers `name="muse_glimmer"` into the `renderers` library. |
| `atem.py` | ATEM (`<atem:function_calls>`) tool-call render/parse. |
| `tools/convert_qk_layout.py` | **Mandatory** one-time offline Q/K RoPE layout conversion of the HF checkpoint. |
| `tools/smoke_generate.py` | Fast "is the checkpoint sane?" generation check (`finish_reason=stop` + valid ATEM call); `--eager` bypasses the nightly torch.compile bug. |
| `standin_retriever.py` | Synthetic `/retrieve` server for a plumbing-only POC (no FAISS index needed). |
| `assets/qa_smoke.parquet` | Tiny QA fixture (question/golden_answers) used by the smoke POC. |
| `tests/` | CPU-testable unit tests (QK-layout parity, renderer round-trip). |

## 🚨 Step 0 (mandatory): convert the checkpoint's Q/K RoPE layout

The released HF export stores `q_proj`/`k_proj` rows in HF's **split-half**
`rotate_half` RoPE layout; TorchTitan's `ComplexRoPE` expects the **interleaved**
layout. The `MuseGlimmerStateDictAdapter` normally applies this permute on load —
but under the RL trainer's **FSDP × TP** sharding the q/k weights are
`_StridedShard` DTensors sharded on the exact dim the permute reshapes, so the
in-adapter permute **cannot** be applied correctly and silently converts only the
generator path, leaving the trainer on the wrong layout → **gibberish rollouts,
every completion truncated, reward 0, and no error raised**.

So convert **once, offline**, on plain tensors, then load with the adapter permute
disabled:

```bash
python -m torchtitan.experiments.rl.examples.glimmer_search_r1.tools.convert_qk_layout \
    /path/to/Muse-Glimmer-30B \
    ./assets/hf/Muse-Glimmer-30B-qk-converted
export GLIMMER_QK_ALREADY_CONVERTED=1   # tells the adapter to load q/k verbatim
```

Point the config's `hf_assets_path` at the converted directory (it already
defaults to `./assets/hf/Muse-Glimmer-30B-qk-converted`).

## Step 1 (recommended): smoke-test the checkpoint

```bash
GLIMMER_QK_ALREADY_CONVERTED=1 torchrun --nproc_per_node=2 \
  -m torchtitan.experiments.rl.examples.glimmer_search_r1.tools.smoke_generate \
  --checkpoint $PWD/assets/hf/Muse-Glimmer-30B-qk-converted --eager
```

Expect `finish_reason: stop`, coherent text, and a parseable ATEM `search` call.
If you instead see `finish_reason: length` + gibberish (e.g. a repetition loop) + no
tool call, the Q/K conversion did not take (re-check Step 0 and
`GLIMMER_QK_ALREADY_CONVERTED`) — this is a verified positive/negative signal.

Notes:
* Pass an **absolute** `--checkpoint` path (the loader rejects relative paths).
* `--eager` disables vLLM cudagraph/torch.compile. On recent torch/vllm nightlies the
  `aot_compile` path can hit an unrelated `WeakValueDictionary` pickling error during KV
  profiling; `--eager` avoids it. Drop `--eager` once your torch/vllm build is past that bug.

## Step 2: start a retriever

Use the real dense retriever (e5 over wiki-18) on `http://127.0.0.1:8000/retrieve`,
or the stand-in server for a plumbing-only POC:

```bash
# POC only: returns synthetic passages (oracle mode leaks golden answers ->
# proves the pipeline, NOT retrieval quality or convergence).
python -m torchtitan.experiments.rl.examples.glimmer_search_r1.standin_retriever \
  --port 8000 --mode oracle --answers-parquet <dir with train.parquet>
```

## Step 3: run RL

```bash
GLIMMER_QK_ALREADY_CONVERTED=1 python -m torchtitan.experiments.rl.train \
  --module glimmer_search_r1 \
  --config rl_muse_glimmer_search_r1 \
  --hf_assets_path=$PWD/assets/hf/Muse-Glimmer-30B-qk-converted
```

For a fast, self-contained plumbing check (2 steps, local QA fixture + stand-in
retriever), use the smoke config instead:

```bash
GLIMMER_QK_ALREADY_CONVERTED=1 python -m torchtitan.experiments.rl.train \
  --module glimmer_search_r1 \
  --config rl_muse_glimmer_search_r1_smoke \
  --hf_assets_path=$PWD/assets/hf/Muse-Glimmer-30B-qk-converted
```

Watch `validation_reward/_mean` on W&B (project set via the config's
`MetricsProcessor`).

**On a single node without InfiniBand**, force NCCL/CTRAN off IB or the TorchStore
weight-sync collective will hang (and CPU-spin):

```bash
export NCCL_CTRAN_BACKENDS=nvl,socket NCCL_IB_DISABLE=1 NCCL_NET=Socket
```

(The RL launcher's GPU provisioner already forwards these `NCCL_*` vars into the
monarch-spawned trainer/generator procs; export them before launching.)

## Hard constraints (verified against the released 30B)

- **Tensor-parallel ≤ 2 everywhere** — Muse Glimmer has **2 KV heads**, so
  `tensor_parallel_degree` must divide 2 on BOTH the vLLM generator AND the trainer
  (the trainer raises `tensor_parallel_degree (N) must divide n_kv_heads (2)` otherwise).
  Shard the rest of the trainer's memory with FSDP: the config uses TP=2 × FSDP=3 = 6
  trainer GPUs + 2 generator GPUs on an 8-GPU node.
- **FullAC required** — Adam m/v states allocate on the *first* `optimizer.step()`;
  the default SelectiveAC (or too few FSDP shards) OOMs at ~92 GB/GPU. TP=2 × FSDP=3
  fits on 96 GB H100s; TP=2 × FSDP=2 (4 GPUs) OOMs.
- **varlen (FA3) attention** on both trainer and generator (needs H100/H200+).
- Start from the **Q/K-converted** checkpoint with `GLIMMER_QK_ALREADY_CONVERTED=1`
  (or the original HF checkpoint with the flag unset, which lets the adapter permute).
- **Fresh dump folder** — a stale DCP checkpoint under `{dump_folder}/checkpoint` will
  trigger a resume that fails with `Missing key ... o_gate`; the smoke config points at a
  clean `dump_folder`.
- **No-InfiniBand nodes**: export `NCCL_CTRAN_BACKENDS=nvl,socket NCCL_IB_DISABLE=1` (see Step 3).

## POC caveats (when using the stand-in retriever)

The stand-in retriever's oracle mode returns passages containing the golden
answer, so reward saturates and only a fraction of steps produce a gradient.
This proves the **end-to-end pipeline**, not convergence or real retrieval
quality. For real training, use a real dense retriever and the strict EM rubric.
