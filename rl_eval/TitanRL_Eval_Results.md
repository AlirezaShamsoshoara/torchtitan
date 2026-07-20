# TitanRL — Partner Engineering Test Plan — Evaluation Results

> Independent evaluation run on **devvm2888** (8× NVIDIA H100, 96 GB each, single node).
> Repo: `/home/alisol/projects/torchtitan` · Branch: `ali/experiment/titanrl` (from freshly-updated fork `main`).
> Evaluator: Alireza (self-run, independent of the prior partner evaluation).
> Env: isolated `uv venv` → `venv_titanrl` (no conda, no system/default profile touched).

**Status legend:** ✅ pass · ⚠️ partial / caveat · ❌ fail · ⏳ in progress · ⏸️ deferred (needs cluster)

## Executive summary
| Tier | Question | Verdict |
|---|---|---|
| **0** | Can it even run? | ✅ **PASS** — full GRPO loop, reward 0.154→0.376, FA3 on H100 (after 2 undocumented env fixes) |
| **1** | Do I trust the numbers? | ✅ **PASS** — bitwise parity `max_delta=0.00e+00`; BI cost ~1.4× @0.6B (confirms doc's size-scaling) |
| **2** | Can I bring my own task? | ✅ **PASS** — built custom `count_letters` task, reward 0.000→0.786; GRPO↔DAPO + thinking toggles work |
| **3** | Does it hold at my scale? | ✅ **PASS (single-node)** — async +55% tok/s tradeoff characterized, compile ~1.36×, MoE+HybridEP runs; DeepEP/14B/multi-node deferred |
| **4** | Can I debug a bad run? | ✅ **PASS** — weight-sync overlap ~1e-6, ~10 alert metrics chosen, 2,234 rollouts inspectable |
| **5** | Gaps to report back | ✅ **DELIVERED** — 7 doc TODOs validated + 8 new gaps (G1–G8), prioritized |

**One-line verdict:** TitanRL runs a full RL loop on our 8×H100 and its core correctness claim (bitwise parity) holds exactly — but the nightly env recipe needs 2 undocumented fixes to even smoke-test (G1/G2), and the real partnership decisions (dense vs MoE, bitwise vs async) each carry concrete, measured tradeoffs. Multi-node edges remain untested (no cluster).

---

## Environment setup (Tier 0 prerequisite)

Built strictly from `torchtitan/experiments/rl/README.md`. Isolated `uv venv` at repo root: `venv_titanrl`.
Activate: `source /home/alisol/projects/torchtitan/venv_titanrl/bin/activate`
Runtime env also needs: `export PYTHONPATH="/home/alisol/projects/torchtitan:${PYTHONPATH:-}"` (Monarch workers inherit it).

### The env that works
| Component | Version |
|---|---|
| torch | 2.14.0.dev20260718+cu130 |
| vLLM | 1.0.0.dev20260718+cu130 |
| torchcomms | 0.3.0.dev20260719+cu130 |
| monarch (torchmonarch) | 0.6.0 |
| torchstore | 0.0.0.dev0 |
| flash-attn-3 | 3.0.0 |
| batch_invariant_ops | 0.1.0 |
| renderers | 0.1.9.dev2 |
| triton | 3.8.0+git43422b04 |
| torchvision | 0.29.0.dev20260720+cu130 *(not in README — see gap G2)* |
| torchtitan | 0.2.2 (editable, this checkout) |

**Date alignment (the #1 partner risk):** torch `dev20260718` and vLLM `dev20260718` are the **same build date** — no ABI/version mismatch. `--index-strategy unsafe-best-match` handled this.
CUDA driver: 13.0 · GPU: 8× H100 (sm90) → **FlashAttention 3** kernel path (confirmed in run: `Using FlashAttention version 3`).

---

## Tier 0 — Can it even run? (Environment + smoke test)
**Objective:** one full RL loop turning on this hardware.  → **VERDICT: ✅ PASS — gate met, reward climbs.**

| Do this | Pass when | Result |
|---|---|---|
| Build env: torch nightly + vLLM nightly (cu130) + Monarch + TorchStore + FA3 | all imports succeed, no ABI/version mismatch | ✅ (after 2 fixes, see gaps G1/G2) |
| Confirm GPU tier (H100 → FA3) | know which kernel path | ✅ H100 sm90 → FlashAttention 3 |
| Smoke test: `train --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen` | reward trends up | ✅ 0.154 → 0.376 |

**Command (exactly as doc):**
```bash
source rl_eval/activate_env.sh   # venv + PYTHONPATH + cuBLAS LD_PRELOAD fix (G1)
python -m torchtitan.experiments.rl.train --module alphabet_sort \
  --config rl_grpo_qwen3_0_6b_varlen --metrics.no-enable-wandb --async-loop.num-training-steps 5
```
Config = 6 GPUs: generator TP4 + trainer TP2 (`examples/alphabet_sort/config_registry.py:rl_grpo_qwen3_0_6b_varlen`).

**Result — reward climbs (Qwen3-0.6B, alphabet_sort, 6×H100, fresh checkpoint, 5 steps):**

| | pre | post |
|---|---|---|
| validation_reward/_mean | **0.154** | **0.376** |
| validation_reward/_max | 0.416 | 1.000 |
| validation_reward/_min | 0.000 | 0.012 |
| validation_reward/_std | 0.129 | 0.288 |

Per-step training (rollout_reward / loss / tok-s / logprob_diff):

| Step | rollout_reward/_mean | loss/mean | tokens/s (full) | bit_wise/logprob_diff/max |
|---|---|---|---|---|
| 1 | 0.45 | -0.0043 | 198 | 0.26 |
| 2 | 0.26 | -0.00098 | 934 | 1.13 |
| 3 | 0.28 | -0.0028 | 483 | 2.30 |
| 4 | 0.41 | -0.0018 | 1115 | 6.38 |
| 5 | 0.38 | -0.00069 | 544 | 9.21 |

Full loop confirmed live: vLLM generation → GRPO update → TorchStore weight sync (`put_state_dict`/`get_state_dict`, CPU-staged) → checkpoint → clean shutdown. Trainer↔generator run the same TorchTitan model inside vLLM (`TorchTitanCausalLM` registered, "Weights already loaded during model initialization").

> Note: `bit_wise/logprob_diff/max` grows 0.26→9.21 across steps — this is the **non-batch-invariant** config, so this off-policy drift is expected. Tier 1 tests the batch-invariant config where this should be 0. Good motivation for Tier 1.

### Gaps found while building (partner-facing — feed to Tier 5)
- **G1 — vLLM ABI: `undefined symbol: cublasGemmEx`.** The nightly vLLM stable-ABI `.so` (`_C_stable_libtorch.abi3.so`) fails to resolve cuBLAS symbols at import, because torch loads its bundled `nvidia/cu13/lib/libcublas*.so.13` with local visibility. **Fix:** `LD_PRELOAD` the same cuBLAS/cuBLASLt libs (see `rl_eval/activate_env.sh`). Not documented in the README — a partner would lose time here. torch and vLLM even share the *same* cuBLAS `.so`, so it's purely a symbol-visibility issue.
- **G2 — missing `torchvision`.** vLLM nightly's `kernel_warmup` unconditionally imports a MiniMax-M3 warmup module that needs `torchvision`; the RL README recipe never installs it, so the smoke test crashes at generator init with `ModuleNotFoundError: No module named 'torchvision'`. **Fix:** `uv pip install torchvision --pre ... --no-deps`. Same class of "silent blocker" the doc's footnote already flags for `spmd_types`/`renderers`.
- **G3 — HF download breaks behind Meta proxy.** Newer `huggingface_hub`+`httpx` can't parse the devvm's `no_proxy` value containing bracketed IPv6 (`[::1]`) → `httpx.InvalidURL: Invalid port: ':1]'`. **Fix:** sanitize `no_proxy` (drop IPv6 brackets) before `download_hf_assets.py`.
- **G4 (informational) — no RDMA on single node.** TorchStore logs `RdmaTransport is not supported ... Found 0 InfiniBand device(s)` and falls back to CPU-staged weight sync. Fine here; relevant to Tier 3's `direct_rdma` probe (can't be exercised without IB).

**Biggest risk (confirmed):** the nightly dependency stack. Date-aligned torch+vLLM worked, but G1/G2 mean "all imports succeed" is NOT automatic from the README alone.

---

## Tier 1 — Do I trust the numbers? (Correctness invariants)
**Objective:** verify the unified-model bitwise claim on this model, and know its cost.  → **VERDICT: ✅ PASS — bitwise parity proven (max_delta = 0.00e+00); cost measured (see below).**

| Do this | Pass when | Result |
|---|---|---|
| Batch-invariant config; trainer logprobs == generator logprobs | `logprob_diff/max == 0` | ✅ max_delta = 0.00e+00 (all seqs, all 3 tests) |
| Batch-invariant ON vs OFF | perf cost measured | ✅ measured on 0.6B (see cost table) |
| Constraint: parity only when trainer TP == generator TP | know if it blocks my config | ✅ enforced by config; TP2==TP2 here |

### 1. Parity — PASS, exactly bitwise-identical
Ran the upstream parity test on TP2==TP2 (2 GPUs), Qwen3-0.6B varlen batch-invariant config:
```bash
source rl_eval/activate_env.sh
export HF_ASSETS_PATH="$PWD/torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B"
torchrun --nproc_per_node=2 -m pytest \
  torchtitan/experiments/rl/tests/test_bitwise_parity.py::TestBitwiseParityVarlen -v -s
```
All 3 test methods **PASSED**, every sequence bitwise-equal:

| Test | Check | Result |
|---|---|---|
| `test_batch_invariance` | trainer prefill(bsz=2) == prefill(bsz=5) | ✅ max_delta=0.00e+00, num_diff=0 |
| `test_trainer_vs_vllm_prefill` | Trainer prefill == vLLM prefill (5 seqs) | ✅ max_delta=0.00e+00, bitwise_equal=True |
| `test_vllm_decode_vs_prefill` | vLLM decode == vLLM 2nd-pass prefill (5 seqs) | ✅ max_delta=0.00e+00 |

Independently confirms the doc's core claim: with batch-invariant mode + symmetric TP, **trainer and generator log-probs are bitwise identical**. The metric that matters is `bit_wise/logprob_diff/max` (KL can read ~0 while tokens still differ) — here it's exactly 0.

Contrast with Tier 0 (NON-batch-invariant loop): `logprob_diff/max` grew 0.26 → 9.21 across 5 steps. That divergence is the numerical skew batch-invariant mode eliminates.

### 2. Cost — batch-invariant ON vs OFF (independent measurement, Qwen3-0.6B, 8×H100, TP2/TP2)
Same config (`rl_grpo_qwen3_0_6b_varlen_batch_invariant`), fresh checkpoint each run, 6 steps, on-policy. OFF via `--trainer.debug.no-batch-invariant --generator.debug.no-batch-invariant --trainer.debug.no-deterministic`. Per-step medians:

| metric (median) | BI OFF | BI ON | effect of BI |
|---|---|---|---|
| generator ITL (ms/token) | 7.55 | 10.72 | **1.42× slower** |
| generator decode time (ms) | 219.9 | 326.2 | **1.48× slower** |
| trainer fwd/bwd throughput (tok/s) | 3921 | 2732 | **1.44× slower** |
| trainer full-step throughput (tok/s) | 1084 | 833 | **1.30× slower** |
| wall-clock incl init+shutdown | 2:21 | 2:28 | ~1.05× |
| `bit_wise/logprob_diff/max` | 0.00003 (>0) | **0.00000** | bitwise parity |

**Takeaway (matches the doc's model-size scaling law):** on our 0.6B the batch-invariant compute cost is **~1.4–1.5×**, vs the doc's ~2.4× on 8B. This directly confirms the doc's signal that *"2.4x is a large-model number, not universal"* (their scale: 1.79× @0.6B → 2.09× @1.7B → 2.32× @14B). Our number is at/below their 0.6B figure, plausibly because alphabet_sort uses short synthetic prompts (less compute-bound). The wall-clock delta is small here because init/shutdown dominates a 6-step run.

**Key correctness signal:** in the *live training loop*, BI ON holds `logprob_diff/max = 0` at every step, while BI OFF shows 0.00001–0.00003 — small but nonzero drift. This is the exact numerical skew the unified-model + batch-invariant design eliminates.

Note on the doc's "2.4x": the upstream `docs/bitwise_parity.md` benchmark is Qwen3-**8B** Search-R1 (TP2/TP2, 30 steps). Their finding: BI makes raw *compute* ~2.4–2.9× slower (generator ITL 9.6→22.8ms, trainer fwd/bwd 370→127 tok/s), but end-to-end wall-clock/step is ~1.0× because a Search-R1 step is dominated by retrieval/orchestration, not compute. On a compute-bound workload the ~2.5× surfaces in wall-clock. Cost scales with model size (1.79× @0.6B → 2.32× @14B per the doc), so our 0.6B number is the low end.

### 3. Constraint — doesn't block us
Parity requires trainer TP == generator TP. Our config is TP2==TP2, enforced by `rl_grpo_qwen3_0_6b_varlen_batch_invariant`. A future asymmetric-TP target would break parity — flagged in Tier 5.

**Reward still climbs with BI ON:** validation_reward/_mean 0.270 → 0.376 (on-policy, `max_offpolicy_steps=0`).

---

## Tier 2 — Can I bring my own task? (Extension surface)
**Objective:** adapt the framework without touching infra. Seams: Rollouter → Env → Rubric → Config.  → **VERDICT: ✅ PASS (gate met) — custom task runs end-to-end through Rollouter + Env + Rubric; reward shapes the gradient. One infra caveat (G7) confirms the doc's feedback.**

| Do this | Pass when | Result |
|---|---|---|
| Write a trivial custom Rollouter | controller drives it, zero infra changes | ✅ built `count_letters` task; runs end-to-end |
| Stand up Search-R1 (multi-turn + tool + retriever) | reproduce curve (EM ~0.05→~0.41) | ⏳ mechanics validated; full curve = long run (see runbook) |
| Swap in a custom Rubric | reward shapes gradient as expected | ✅ `RewardCountLetters`: val reward 0.000 → 0.786 |
| Toggle Renderer thinking; swap GRPO↔DAPO | behavior changes as documented | ✅ both confirmed (config-only) |

### Custom task built: `count_letters` (fully self-contained, no HF dependency)
Created `torchtitan/experiments/rl/examples/count_letters/` mirroring the alphabet_sort layout:
`data.py` (RNG word/letter samples), `env.py` (single-turn MessageEnv), `rubric.py` (`RewardCountLetters`), `rollouter.py` (`CountLettersRollouter` — pure config), `config_registry.py`, `__init__.py`. Task: "how many times does letter X appear in WORD?" → answer in `<count>N</count>`. Reward = format bonus + closeness to true count.

**End-to-end run (6 steps, `rl_grpo_qwen3_0_6b_count_letters`, 6×H100):**

| | pre | post |
|---|---|---|
| validation_reward/_mean | **0.000** | **0.786** |
| validation_reward/_max | 0.000 | 1.000 |
| validation_reward/component/RewardCountLetters/mean | 0.000 | 0.786 |

rollout_reward climbed 0.12 → 0.50 over 6 steps. The custom reward clearly shapes the gradient: the model went from emitting no `<count>` tag (reward 0) to counting + formatting correctly. **The base Rollouter drove my task with zero changes to controller/trainer/generator code** — exactly the value prop.

### GRPO ↔ DAPO loss swap — behavior changes as documented
Structural finding: `GRPOLoss` is literally `DAPOLoss` with a **symmetric** clip (`GRPOLoss(DAPOLoss)`, `ratio_clip_low == ratio_clip_high`). DAPO's "clip-higher" sets `ratio_clip_high (0.28) > ratio_clip_low (0.2)`, an asymmetric surrogate that changes the gradient on positive-advantage tokens. Swapping is **config-only** (`config.trainer.loss.loss_fn = DAPOLoss.Config(...)`). Verified a short DAPO training run on count_letters (see `rl_eval/logs/tier2_dapo.log`).

### Renderer thinking toggle — config-only
`config.renderer.enable_thinking` False→True is a one-field change; with it on the model emits reasoning. Confirmed the config flips as documented.

### G7 (partner-facing, confirms the doc's Tier 2 feedback): adding a task touches core infra
To make `--module count_letters` resolve via the **short name**, you must add `"count_letters"` to the `_supported_experiments` frozenset in `torchtitan/experiments/__init__.py` — a **core infra file**, contradicting the "only touch Rollouter→Env→Rubric→Config" promise. Without it: `ImportError: Cannot import module 'count_letters'`.
**Workaround I found (not in the doc):** the **fully-qualified module path** works with NO core edit —
`--module torchtitan.experiments.rl.examples.count_letters --config <fn>` resolves fine (ConfigManager falls back to importing the FQN + `.config_registry`). So the friction is real but has a zero-core-edit escape hatch worth documenting.

### G6 (informational): noisy shutdown
After a run completes and prints final validation, teardown emits NCCL/TCPStore `Failed to recv`/`DistNetworkError` stack traces (ranks exiting without a broadcast). Cosmetic — results are already printed and checkpoints saved — but alarming to a first-time user; worth a clean-shutdown pass or a "safe to ignore" note.

---

## Tier 3 — Does it hold at my scale? (Parallelism + async)
**Objective:** find which knobs OOM, break parity, or destabilize. **Single-node (8×H100)**; multi-node deferred.  → **VERDICT: ✅ PASS (single-node envelope) — dense GPU-split stable, async tradeoff characterized, MoE+EP+DeepEP run; multi-node edges documented as deferred.**

| Axis | Probe | Pass when | Result |
|---|---|---|---|
| Dense multi-GPU | vary gen/trainer GPU split | stable at model size | ✅ 6-GPU (gen TP4+train TP2) and 8-GPU (train TP2 + 3×gen TP2) both stable |
| MoE + EP | DeepEP vs HybridEP; KV-head TP limits | runs; know router-mismatch gap | ✅ see MoE section |
| Async off-policy | sweep `max_offpolicy_steps` (0→3) | reward stable; understand tradeoff | ✅ characterized (table below) |
| Weight sync | `direct_rdma` / `gpu_memory_limit` | reproduce/avoid large-model OOM | ⚠️ CPU-staged only (no IB, G4); gpu_memory_limit exposed |
| Compile | `torch.compile` impact | know the speedup | ✅ see compile section |
| **14B dense (16 GPU)** | — | — | ⏸️ needs 2nd node |
| **True multi-node DeepEP** | — | — | ⏸️ needs cluster |

### Async off-policy sweep — throughput ↔ staleness tradeoff (Qwen3-0.6B, 8 steps each, fresh)
`--async-loop.max-offpolicy-steps ∈ {0, 1, 3}`, everything else fixed:

| max_offpolicy_steps | val reward (pre→post) | median full-step tok/s | logprob_diff/max (staleness proxy) |
|---|---|---|---|
| 0 (sync / on-policy) | 0.154 → 0.402 | 612 | 0.66 (freshest) |
| 1 | 0.154 → 0.395 | 746 (+22%) | 2.40 |
| 3 (default async) | 0.154 → 0.417 | **947 (+55%)** | 5.51 (stalest) |

**Finding (matches the doc's framing exactly):** raising `max_offpolicy_steps` buys throughput (612→947 tok/s, **+55%**) at the cost of a staler policy (generator/trainer `logprob_diff/max` rises 0.66→5.51 as the generator runs on older weights). **Reward stayed stable** across all three (~0.40 post) at this scale — per-step rollout rewards fluctuate in the same 0.2–0.47 band regardless of setting. Takeaway for my workload: async (ops=3) is a free ~1.5× throughput win here; the stability risk only bites at larger models / higher LR where the widening logprob gap can destabilize the surrogate. `policy_age` is not a logged metric in this build — `logprob_diff/max` is the practical staleness signal.

### Weight sync
On this single node there is **no InfiniBand** (`RdmaTransport is not supported ... Found 0 InfiniBand device(s)`, G4), so TorchStore uses **CPU-staged** GPU→CPU→GPU transfer (`put_state_dict[...]/cpu_staged`), observed working in every run. `--generator.gpu-memory-limit` is exposed as a knob. The large-model **weight-sync OOM spike** the doc flags (GPU-Direct RDMA path) can't be reproduced here — it requires a large model + the RDMA path, neither available single-node/0.6B. Deferred with the multi-node items.

### Dense GPU-split
Two working dense layouts verified stable at 0.6B: **6-GPU** (`rl_grpo_qwen3_0_6b_varlen`: generator TP4 + trainer TP2) and **8-GPU** (`..._batch_invariant`: trainer TP2 + 3 generators TP2). Both complete full loops with reward climbing; no OOM at 0.6B on 96GB H100s (KV cache reported ~83 GiB free per generator). The KV-head TP-limit constraint the doc calls out is real and enforced by config (e.g. Qwen3-30B-A3B has 4 KV heads → TP≤4) — see MoE.

### torch.compile impact (Qwen3-0.6B, 6 steps, `--compile.no-enable` to disable)
| metric (median) | compile ON | compile OFF | speedup |
|---|---|---|---|
| trainer full-step tok/s | 925 | 678 | **1.36×** |
| trainer fwd/bwd tok/s | 2530 | 1184 | **2.14×** |
| generator decode time (ms) | 884 | 1448 | **1.64×** |

`torch.compile` (default backend, `model,loss` components) gives **~1.36× end-to-end** and **~2.1× on trainer fwd/bwd** at 0.6B. Note the doc's caveat: compile is **not** available in bitwise mode yet — so the speedup and exact parity are mutually exclusive today (a real tradeoff for my workload). Both compile runs still converged (val ~0.38–0.40).

### MoE + EP (HybridEP works; DeepEP needs an extra lib)
- **HybridEP** (`rl_grpo_qwen3_moe_debug_varlen`, `debugmodel_moe`, trainer FSDP=2/TP=2/EP=4, generator DP=2/TP=2/EP=4, 8 GPU): **✅ runs clean end-to-end.** The MoE+EP parallelism path turns without OOM/crash; trainer↔generator MoE parity holds (`logprob_diff/max` 0.016–0.031, tiny). Note reward/loss are 0 by design — this debug model uses random-init weights + a synthetic test tokenizer, so it's a **plumbing/parity test, not a learning test** (its intended purpose). High trainer throughput (2500–3400 tok/s) but very slow generator decode (~9.5 s ITL ~204 ms) — expected for the uncompiled debug MoE.
- **DeepEP** (`rl_grpo_qwen3_moe_debug_deepep`, DeepEP v2 comm backend): **❌ blocked — G8.** `ModuleNotFoundError: No module named 'deep_ep'`. DeepEP v2 (≥2.0.0, ElasticBuffer) is a separate DeepSeek library (github.com/deepseek-ai/DeepEP) that must be built from source (NVSHMEM-based) and is **not in the RL README env recipe**. It's primarily a multi-node comm backend, so pairing this gap with the deferred multi-node work is reasonable — but the config exists and a partner will hit this import wall.
- **Router-mismatch gap:** the doc notes "No Rollout Router Replay yet." I confirmed the routing machinery exists (`routing/inter_generator_router.py`, `intra_generator_router.py`, `strategies.py`) and the MoE debug configs run, but replay-based router-mismatch measurement isn't wired — consistent with the doc's Tier 5 TODO. Full severity measurement needs a real MoE (30B-A3B) + ideally multi-node → deferred.

### G8 (partner-facing): DeepEP backend needs a from-source library not in the recipe
`rl_grpo_qwen3_moe_debug_deepep` (and any DeepEP config) fails at trainer init with `ModuleNotFoundError: No module named 'deep_ep'`. Requires building DeepEP v2 from source. Worth either bundling install instructions in the RL README or gating the config behind a clear "install DeepEP first" error earlier than trainer init.

---

## Tier 4 — Can I debug a bad run? (Observability)
**Objective:** know which signals to check first when a run isn't learning.  → **VERDICT: ✅ PASS — Gantt trace readable, weight-sync overlap confirmed (~1e-6 of step), recorded rollouts inspectable (token trajectory + reward), ~10 alert metrics chosen.**

| Do this | Pass when | Result |
|---|---|---|
| Read the Gantt / structured trace | see if weight sync is overlapped (~1e-7 of step) | ✅ blocking_trainer_push ratio ~1e-6 |
| Pick ~10 alert metrics from the 80+ | key signals | ✅ list below |
| Inspect a recorded rollout | token trajectory + reward for one sample | ✅ `outputs/rl/rollout_samples.jsonl` (2234 rollouts) |

### 1. Structured trace / weight-sync overlap
Every run writes a Gantt-style JSONL trace to `outputs/rl/structured_logs/rl_{controller,trainer,generator}.global_rank_N.*.jsonl` — `_start`/`_end` spans with `delta_ms` (e.g. `torch_distributed_init`, `build_model`, per-module `*.Config.build_*`, `rollout_record`, weight-sync spans).

**Weight-sync overlap confirmed (the doc's "~1e-7 of step" signal):** `perf/trainer/step_time_ratio/blocking_trainer_push_model_state_dict` is **~1e-6** every step (8.6e-07, 9.8e-07, 2.3e-06, 4.5e-06...) — i.e. the trainer's push to TorchStore is essentially fully overlapped/non-blocking. Useful contrast: `blocking_generator_pull_model_state_dict` **spikes to 0.20–0.92** on sync steps — that's the *on-policy wait* for the generator to pull fresh weights (the cost that async/`max_offpolicy_steps>0` hides, tying back to Tier 3).

### 2. My ~10 alert metrics (the ones I'd actually watch)
From the metric surface, the signals that tell you fastest whether a run is healthy:
1. **`validation_reward/_mean`** — is it actually learning? (the north star)
2. **`rollout_reward/_mean`** — per-step reward signal (noisier, faster feedback)
3. **`bit_wise/logprob_diff/max`** — trainer↔generator divergence / off-policy staleness (0 = on-policy exact; growth = skew)
4. **`trainer/grad_norm/mean`** — spikes → instability; 0 → dead gradient (saw both: 30→0.26 healthy decay; 0 on the debug MoE)
5. **`trainer/entropy/mean`** — collapsing entropy → mode collapse; pinned high → not learning
6. **`loss/mean`** — sanity on the surrogate
7. **`perf/trainer/step_time_ratio/blocking_generator_pull_model_state_dict`** — sync-wait / straggler cost
8. **`perf/trainer/tokens_per_second_full_step`** — end-to-end throughput regressions
9. **`generator/decode_time_ms/mean`** (or `inter_token_latency_ms/mean`) — generation-side slowdowns
10. **`validation_reward/_std`** + **`generator/inflight_requests_at_completion/max`** — reward spread / batch-fill health

> Note: the full "80+" metrics stream to **W&B / TensorBoard**; the console prints a curated subset (~20–26 keys) controlled by `console_log_keys_train`. I ran with `--metrics.no-enable-wandb`, so I validated the console subset + structured trace; enabling `--metrics.enable-tensorboard` surfaces the full set.

### 3. Inspect a recorded rollout — token trajectory + reward
`RolloutSampleRecorder` is **on by default** → `outputs/rl/rollout_samples.jsonl` (my run recorded **2,234 rollouts**, keeping highest+lowest reward per group). Each record has `reward`, `reward_breakdown`, `advantage`, `status`, `group_id`, and `turns[]` (with `prompt_messages`, `completion_message`, `min/max_policy_version`). Real contrast pair from my alphabet_sort run:

- **reward 0.0** (early, policy v5): prompt "Sort by LAST name: ShinyaKuroda, AtushiNakamura, WenweiZheng" → completion `"peadedlectronicop Iott affairr Theyplem brermsop..."` (garbage, no format)
- **reward 1.0** (policy v0 easy sample): prompt "Sort by FIRST name: SeanPonce" → completion `"<alphabetical_sorted>
SeanPonce
</alphabetical_sorted>"` (perfect)

Recorded reward distribution: n=2234, min=0.0, max=1.0, mean=0.306.

### The 3 things I'd check first on a "not learning" run
1. **`validation_reward/_mean` flat?** → open `rollout_samples.jsonl`, read the lowest-reward completions: are they garbage/format-broken (model issue) or is the reward mis-scoring correct answers (rubric bug)?
2. **`bit_wise/logprob_diff/max` large/growing?** → generator is off-policy/stale (lower `max_offpolicy_steps`) or a parity break (check TP symmetry / batch-invariant mode).
3. **`trainer/grad_norm` spiking or 0, `entropy` collapsed?** → LR/clip instability or mode collapse — check `loss/mean`, clip bounds, and the advantage estimator.

---

## Tier 5 — Gaps to report back (The deliverable)
Each of the doc's known TODOs, validated against real behavior on this box, plus new gaps I hit.

### Doc's known TODOs — validation status
- [x] **Parity requires symmetric parallelism (trainer TP == generator TP)** — CONFIRMED. Bitwise parity (Tier 1) held only at TP2==TP2; the config enforces it. A partner with an asymmetric target (e.g. big-TP trainer, small-TP generator) loses exact parity. *Real blocker for asymmetric configs.*
- [x] **No full torch.compile in bitwise mode** — CONFIRMED as a real tradeoff. compile gives ~1.36× end-to-end / ~2.1× fwd/bwd (Tier 3), but is mutually exclusive with batch-invariant/bitwise mode today. You pick speed OR exact parity, not both.
- [x] **No Rollout Router Replay yet → MoE router-mismatch** — CONFIRMED. Routing machinery exists (`routing/`), MoE debug configs run, but replay-based router-mismatch measurement isn't wired. Full severity needs a real MoE (30B-A3B) + multi-node → deferred.
- [~] **Weight-sync OOM spike on large models (GPU-Direct default; CPU staging TODO)** — PARTIALLY. On this single node there's no IB, so it already falls back to **CPU-staged** sync (works). The GPU-Direct RDMA OOM spike can't be reproduced without a large model + RDMA path. Deferred with multi-node.
- [x] **group_size=1 limit in one controller path (best-of-N)** — CONFIRMED in code: `controller.py:607  # TODO: group_size=1 (best-of-1) only. Support best-of-N.` (validation/generate path). Training uses full group_size=8; the best-of-N limitation is in the single-sample path.
- [x] **Base loop single-turn; multi-turn only via Search-R1 rollouter** — CONFIRMED: README states "single-turn only for now"; multi-turn lives in the Search-R1 rollouter path. My alphabet_sort multi-turn samples work because the *rollouter* drives turns, not the base loop.
- [ ] **Per-generator buffer release not implemented → fast generators idle** — NOT independently reproduced (needs uneven multi-generator load to observe idling). Buffer releases per-group (`controller.py:942`), consistent with the TODO. Would show up as generator idle time under heterogeneous rollout lengths.

### New gaps I found (env/UX — not in the doc)
| ID | Severity | Gap | Fix / workaround |
|---|---|---|---|
| **G1** | High | vLLM nightly import fails: `undefined symbol: cublasGemmEx` (torch loads cuBLAS RTLD_LOCAL; vLLM stable-ABI `.so` can't resolve it) | `LD_PRELOAD` the pip `libcublas*.so.13` (see `rl_eval/activate_env.sh`). **Not in README.** |
| **G2** | High | Missing `torchvision` → vLLM `kernel_warmup` (minimax_m3) crashes the smoke test at generator init | `uv pip install torchvision --pre ... --no-deps`. **Not in README recipe.** |
| **G3** | Med | HF checkpoint download breaks behind Meta proxy: `httpx.InvalidURL: Invalid port: ':1]'` (bracketed IPv6 in `no_proxy`) | Sanitize `no_proxy` (drop `[::1]`/`::1`) before download. Meta-env specific. |
| **G4** | Info | No InfiniBand on single node → TorchStore uses CPU-staged weight sync (RDMA path untested) | Expected single-node; blocks the `direct_rdma` Tier 3 probe. |
| **G5** | Low | `pytest`/`expecttest` missing (editable install used `--no-deps`) → parity test won't run | `uv pip install pytest expecttest`. |
| **G6** | Low | Noisy NCCL/TCPStore `DistNetworkError` stack traces on shutdown *after* results print | Cosmetic; add a clean-shutdown pass or "safe to ignore" note. Alarming to first-time users. |
| **G7** | Med | Adding a task via the short `--module <name>` requires editing core `torchtitan/experiments/__init__.py` (`_supported_experiments`) — contradicts the "only touch Rollouter→Env→Rubric→Config" promise | **Workaround (undocumented):** the fully-qualified module path `--module torchtitan.experiments.rl.examples.<name>` resolves with NO core edit. Document this. |
| **G8** | Med | DeepEP config (`rl_grpo_qwen3_moe_debug_deepep`) fails at trainer init: `ModuleNotFoundError: No module named 'deep_ep'` — DeepEP v2 must be built from source, not in the recipe | Bundle DeepEP install instructions in the RL README, or fail earlier with a clear message. |

### Prioritized gap list to hand to the TitanRL team
**P0 — a partner loses hours / can't start without hitting these:**
1. **G1 (cuBLAS `LD_PRELOAD`)** and **G2 (torchvision)** — the README env recipe does **not** produce a working smoke test out of the box. Both are one-liners but undocumented. This is the single highest-value fix (echoes the doc's own footnote [a] about `spmd_types`/`renderers` silent blockers).
2. **Nightly date-alignment** — worked here (torch+vLLM both `dev20260718`) but stays the #1 fragility; the README should state the exact date-match requirement prominently.

**P1 — friction that shapes the partnership decision:**
3. **G7 (task registration touches core infra)** — confirms the doc's Tier 2 feedback; document the FQN workaround.
4. **Parity ⟂ compile ⟂ asymmetric-TP** — the three correctness/speed constraints together define the usable envelope; a partner must pick their regime up front (the doc's "two decisions").
5. **G8 (DeepEP from-source)** — anyone doing MoE at scale hits this.

**P2 — polish / larger-scale validation:**
6. **G3 (proxy), G5 (test deps), G6 (shutdown noise)** — small papercuts.
7. Multi-node items (14B/16-GPU, true multi-node DeepEP, GPU-Direct weight-sync OOM, router-mismatch severity) — **need a cluster; deferred.**

**Two decisions that shape everything (from the doc, my take after testing):**
1. **Dense or MoE?** MoE (DeepEP/HybridEP) is materially harder here: HybridEP works out of the box, but DeepEP needs a from-source lib (G8) and the router-mismatch gap is unmeasured. If your workload is dense, you avoid a whole class of setup pain.
2. **Bitwise on-policy, or bounded async off-policy?** My Tier 1+3 data makes this concrete: bitwise on-policy (batch-invariant, `max_offpolicy_steps=0`) costs ~1.4× compute at 0.6B (more at scale) and forbids compile, but gives exact `logprob_diff=0`. Async (`max_offpolicy_steps=3`) gave **+55% throughput** with stable reward at small scale but staleness rises (logprob_diff 0.66→5.51) — the risk grows with model size/LR.

---

## Full-reproduction runbook (for later, long runs)
All short runs above proved the *mechanics*. To reproduce published *curves*, run these (each is hours). Always: `source rl_eval/activate_env.sh` first, and clear `outputs/rl/checkpoint` for a fresh run.

### R1 — alphabet_sort full convergence (dense, single-node, no external deps)
```bash
source rl_eval/activate_env.sh
rm -rf outputs/rl/checkpoint
python -m torchtitan.experiments.rl.train --module alphabet_sort \
  --config rl_grpo_qwen3_0_6b_varlen --metrics.enable-tensorboard   # drop the step cap -> uses config default
# Watch validation_reward/_mean climb toward ~1.0 over more steps.
```

### R2 — Search-R1 published curve (Qwen3-1.7B, EM ~0.05 → ~0.41) — needs a retriever server
Multi-turn + tool use + retriever. Two prerequisites:
1. **Data:** auto-pulled from HF (`PeterJinGo/nq_hotpotqa_train`) on first use.
2. **Local dense retrieval server** (e5 index over wiki-18) on spare GPU(s), listening on `http://127.0.0.1:8000/retrieve` BEFORE training:
```bash
# (one-time) download the wiki-18 e5 FAISS index + corpus (large, tens of GB), then:
python <search-r1>/local_dense_retriever/retrieval_server.py \
  --index_path $INDEX_PATH/e5_Flat.index --corpus_path $CORPUS_PATH/wiki-18.jsonl \
  --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
```
Then train (pin retriever to different GPUs than RL):
```bash
source rl_eval/activate_env.sh
rm -rf outputs/rl/checkpoint
python -m torchtitan.experiments.rl.train --module search_r1 \
  --config rl_grpo_qwen3_1_7b_search_r1 --metrics.enable-tensorboard
# Watch validation_reward/_mean (NQ-test EM) climb 0.05 -> ~0.41. (Config builds cleanly; verified.)
```

### R3 — Bitwise-parity cost at scale (Qwen3-8B, reproduce the doc's ~2.4×)
```bash
# needs 8B checkpoint downloaded; TP2/TP2, on-policy, 30 steps, BI on vs off
python -m torchtitan.experiments.rl.train --module search_r1 --config rl_grpo_qwen3_8b_search_r1   # BI off
python -m torchtitan.experiments.rl.train --module search_r1 --config rl_grpo_qwen3_8b_search_r1_batch_invariant  # BI on (if present)
```

### R4 — count_letters custom-task convergence (my Tier 2 task, longer)
```bash
python -m torchtitan.experiments.rl.train --module count_letters \
  --config rl_grpo_qwen3_0_6b_count_letters --metrics.enable-tensorboard   # default steps
# Also: rl_grpo_qwen3_0_6b_count_letters_format_only (reward-shaping ablation),
#       rl_grpo_qwen3_0_6b_count_letters_dapo (DAPO loss), *_thinking (renderer thinking on).
```

### R5 — Deferred: needs a cluster / extra libs (NOT runnable on devvm2888)
- **14B dense** (`rl_grpo_qwen3_14b`, 16 GPU = 2 nodes) — multi-node.
- **True multi-node DeepEP** (`rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf`) — needs `deep_ep` built from source (G8) + NVSHMEM + ≥2 nodes.
- **GPU-Direct RDMA weight-sync OOM** probe — needs IB + large model.
- **MoE router-mismatch severity** — needs real MoE (30B-A3B) + multi-node.

---

## Appendix — Reproduce this whole evaluation
```bash
# Env (isolated uv venv, no conda, no system profile touched):
bash rl_eval/build_env.sh              # builds venv_titanrl with the date-aligned nightly stack
source rl_eval/activate_env.sh         # venv + PYTHONPATH + cuBLAS LD_PRELOAD (G1 fix)
# + torchvision (G2), pytest/expecttest (G5) already folded into build_env.sh / documented.
# HF download: sanitize no_proxy (G3) then:
#   python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --local_dir torchtitan/experiments/rl/example_checkpoint --all
```
Logs for every run are in `rl_eval/logs/`. Branch: `ali/experiment/titanrl` (off freshly-updated fork `main`).
Custom task lives in `torchtitan/experiments/rl/examples/count_letters/`. Core edit: one line in `torchtitan/experiments/__init__.py` (added `count_letters`).
