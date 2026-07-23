# TitanRL vs. Slime vs. Miles — RL Post-Training Framework Comparison

> **Author:** Alireza Shamsoshoara — independent technical review
> **Date:** 2026-07-20
> **Scope of this document:** A theory-and-implementation comparison of three
> open-source RL post-training stacks for LLMs — **TitanRL** (the `experiments/rl`
> subproject inside PyTorch **torchtitan**), **Slime** (THUDM), and **Miles**
> (RadixArk) — across scope, cross-functional (XFN) footprint, target audience,
> implementation/architecture, feature surface, and performance characteristics.
>
> **Method:** All three repos were read directly (source, READMEs, docs, configs)
> on devvm2888. Slime & Miles were shallow-cloned from GitHub; TitanRL was read
> from the local torchtitan checkout at `torchtitan/experiments/rl/`. No GPU runs
> were executed for this comparison (the TitanRL perf numbers cited here come from
> a prior 8×H100 evaluation on this same box, recorded in `rl_eval/`).
>
> **Import note:** This is a plain Markdown file intended to be pasted/imported
> into a Google Doc. Tables and headings map cleanly to Docs formatting.

---

## 0. TL;DR — one-paragraph each

- **TitanRL** is a *reference RL loop* living inside **PyTorch torchtitan**. Its
  thesis is **correctness through a single, unified model definition**: the exact
  same torchtitan model class is used by both the trainer (torchtitan/FSDP) and
  the generator (vLLM), enabling **bitwise-identical** train/inference logprobs.
  It is small (~12K LOC), research-grade, single-node-friendly, and built on
  Meta's **Monarch** (actor orchestration) + **TorchStore** (RDMA weight sync) +
  **vLLM**. Think "the correctness-first, hackable RL substrate for the PyTorch
  ecosystem."

- **Slime** is a *production-validated, SGLang-native* RL post-training framework
  from **THUDM / Z.ai** (the GLM team). Its thesis is **two capabilities that
  reinforce each other**: high-performance training (Megatron) + flexible data
  generation (SGLang), joined by a **Data Buffer**. It is the RL framework behind
  the **GLM-4.5 → GLM-5.2** model family — battle-tested at frontier scale
  (MoE up to ~355B), Apache-2.0, ~29K LOC, ~6.6K GitHub stars.

- **Miles** is an **enterprise-grade fork of Slime** by **RadixArk** (the
  commercial SGLang company, $100M seed). It keeps Slime's architecture and
  co-evolves with it, but adds a large layer of **systems optimizations for
  large-scale/MoE/low-precision production**: unified **FP8** train+rollout,
  **INT4 QAT**, **Rollout Routing Replay (R3)**, speculative RL, zero-copy weight
  sync, LoRA, an experimental **FSDP backend**, and a custom load-balancing
  **router**. ~46K LOC (a superset of Slime), Apache-2.0.

**The single most important framing:** these are **not three peer competitors**.
Slime and Miles are the *same lineage* (Miles ⊃ Slime, SGLang+Megatron, Chinese
frontier-lab / enterprise origin). **TitanRL is the outsider** — a different
ecosystem (PyTorch-native, vLLM, Monarch), a different primary goal
(numerical correctness / reference implementation rather than max-scale
production throughput), and a much smaller, younger, experimental codebase.

---

## 1. Identity & Provenance

| | **TitanRL** | **Slime** | **Miles** |
|---|---|---|---|
| **Full name** | torchtitan RL experiment (`experiments/rl`) | slime | Miles |
| **Owner / origin** | Meta / PyTorch team | THUDM (Tsinghua) + Z.ai (Zhipu, GLM team) | RadixArk (commercial SGLang company) |
| **Repo** | `pytorch/torchtitan` → `torchtitan/experiments/rl` | `THUDM/slime` | `radixark/miles` |
| **Relationship** | Independent (PyTorch ecosystem) | Upstream / original | **Fork of Slime**, "co-evolving" |
| **License** | BSD 3-Clause | Apache-2.0 | Apache-2.0 |
| **Maturity signal** | "Under active development; APIs may change" (experiment) | "One of the most battle-tested open RL frameworks"; behind GLM-4.5→5.2 | "Enterprise-ready"; behind RadixArk platform |
| **Approx. core LOC (Python)** | ~12,000 | ~29,300 | ~46,500 |
| **GitHub traction** | Part of torchtitan (not standalone) | ~6.6K stars, Apache, large ecosystem | ~1.6K stars, 267 forks (younger) |
| **Headline positioning** | "Unified model definition for bitwise-correct RL" | "LLM post-training framework for RL scaling" | "Enterprise-grade RL for large-scale model training" |

**Key provenance facts**
- Slime is explicitly the post-training stack open-sourced *with* GLM-5.2 and used
  for the whole GLM-4.5/4.6/4.7/5/5.1/5.2 line. It is SGLang-native by deliberate
  design decision (one rollout backend, optimized deeply, rather than a
  lowest-common-denominator multi-backend abstraction).
- Miles' own README: *"Built as a powerful fork of slime… bridges the gap between
  research-grade RL and production-grade reliability."* RadixArk (Accel-backed,
  $100M seed, carries the SGLang/LMSys lineage) maintains Miles as the RL half of
  its end-to-end platform (SGLang = inference half). Miles even has a
  `miles_diffusion` downstream fork of itself.
- TitanRL is a *subdirectory experiment* inside torchtitan — it is not a
  standalone product. Its reason to exist is to prove out an RL loop on the
  canonical PyTorch training stack (torchtitan) with a correctness-first thesis.

---

## 2. Scope — what each one actually is

### TitanRL — *a correctness-first reference RL loop*
- **Narrow, deep, and opinionated.** One trainer backend (**torchtitan**, i.e.
  PyTorch-native FSDP2/TP/PP/CP/EP) + one generator (**vLLM**), joined by
  **Monarch** actors and **TorchStore** weight sync.
- Ships **GRPO** and **DAPO** losses (GRPO = DAPO with symmetric clip). Single-turn
  base loop; multi-turn + tool-use available through the **Search-R1** example
  rollouter.
- Two headline examples: `alphabet_sort` (toy "sum digits / sort" smoke task) and
  `search_r1` (multi-turn retrieval-augmented QA with EM reward).
- Central selling point is the **unified model definition** → **bitwise parity**
  between generator and trainer logprobs, plus a **batch-invariant mode** for
  run-to-run determinism.
- Supported model configs today: Qwen3 (0.6B / 1.7B / 8B / 14B), Qwen3-MoE
  (30B-A3B), and GPT-OSS-20B — via vLLM registration of torchtitan model specs.

### Slime — *a general RL-scaling substrate (train + generate)*
- **Two first-class capabilities:** (1) high-performance **training** (Megatron
  pass-through) and (2) **flexible data generation** (SGLang + custom rollout
  functions), joined by a **Data Buffer**.
- Deliberately **one rollout backend (SGLang)** to exploit SGLang-specific
  features (router, PD-disaggregation, caching, partial rollout) directly.
- Broad algorithm/workflow surface: PPO-style, GRPO, SFT, on-policy distillation,
  agentic multi-turn, tool use, sandboxes, verifiers, multi-agent — all as
  *data-generation* plug-ins that do **not** fork the training kernel.
- Explicitly a **substrate**: a whole ecosystem is built *on* Slime (Miles, vime,
  Dressage, Relax, P1, RLVE, TritonForge, APRIL, OpenClaw-RL, …).
- Models: GLM family, Qwen2.5/3/3-Next/3-MoE/3.5/3.6, DeepSeek V3/V3.1/R1, Llama 3.

### Miles — *Slime + enterprise/production systems layer*
- **Everything Slime does**, plus a heavy **systems-optimization** layer aimed at
  the pain points of *large-scale, MoE, low-precision, production* RL.
- Net-new modules vs. Slime: a `router/` (custom FastAPI load-balancer with health
  checks / quarantine), a `true_on_policy/` package (structured contracts for
  train↔rollout parity), an **experimental FSDP training backend**
  (`backends/experimental/fsdp_utils`) alongside Megatron, LoRA support, and many
  more `examples/` (formal math, SWE-agent, retool-v2, eval harnesses, low
  precision, reproducibility, p2p weight transfer).
- Marquee features: **Unified FP8** (same quantization in rollout & train),
  **INT4 QAT** (1TB models on a single H200), **R3 routing replay**, **speculative
  RL with online-SFT draft model**, **zero-copy CUDA-IPC weight sync**.

**Scope summary:** TitanRL = *narrow + correctness*; Slime = *broad + battle-tested
substrate*; Miles = *broad + enterprise systems hardening (superset of Slime)*.

---

## 3. Architecture & Implementation

### 3.1 Component stack (the defining difference)

| Layer | **TitanRL** | **Slime** | **Miles** |
|---|---|---|---|
| **Training backend** | **torchtitan** (PyTorch-native: FSDP2, TP, PP, CP, EP) | **Megatron-LM** (arg pass-through) | **Megatron-LM** + **experimental FSDP** backend |
| **Inference / rollout** | **vLLM** | **SGLang** (+ sglang-router) | **SGLang** (+ Miles router, deeper integration) |
| **Orchestration** | **Monarch** actors (Meta's distributed actor framework) | **Ray** placement groups + actor groups | **Ray** (same as Slime) |
| **Weight sync** | **TorchStore** (GPU→GPU RDMA when available) | Distributed (NCCL broadcast) / disk / disk-delta / tensor | + **zero-copy CUDA-IPC**, bucketed, async gather, p2p |
| **Model definition** | **Unified** — one torchtitan model class for BOTH train & infer | Separate: Megatron model for train, SGLang model for infer, bridged via mbridge / megatron_to_hf | Same as Slime + FP8/INT4 aligned conversion paths |
| **Data path** | `data → rollout → batcher → train` async loops w/ work buffer | `rollout → Data Buffer → train` (Ray) | Same as Slime + partial-rollout / over-sample recycling |
| **Loss / algo** | GRPO, DAPO (in-repo) | PPO/GRPO/SFT/distillation via `ppo_utils` + rm_hub | Same + DrGRPO, TIS/MIS off-policy correction |
| **Language / config** | Python dataclass configs (`tyro`), `config_registry.py` | CLI args: `--` (slime) / `--sglang-` / raw Megatron | Same convention + `cli-reference` doc, arg groups |

The **single biggest architectural distinction**: TitanRL uses **one model
definition** shared by trainer and generator (torchtitan class registered into
vLLM). Slime/Miles keep **two** model stacks (Megatron for training, SGLang for
inference) and bridge them via weight conversion (`megatron_to_hf`, `mbridge`).
That design choice is exactly what lets TitanRL claim *bitwise* generator/trainer
parity as a native property, whereas Slime/Miles must *engineer* parity (Miles'
FP8-unify + R3 routing replay + TIS/MIS are all about closing the
train-inference mismatch that a two-stack design creates).

### 3.2 TitanRL control flow (from `controller.py`)

```diagram
_data_input_loop --> _rollout_loop[N] --> _batcher_loop --> training_batch_queue --> _trainer_loop
        |                    ^                    ^                                         |
        v                    |                    |                                         |
        +-------------- RolloutGroupWorkBuffer ---+-----------------------------------------+
                        (active slots = (max_offpolicy_steps + 1) * num_groups_per_train_step)
```

- Fully **async** pipeline of asyncio loops; back-pressure is governed by a
  `RolloutGroupWorkBuffer` whose slot count = `(max_offpolicy_steps + 1) *
  num_groups_per_train_step`.
- `max_offpolicy_steps = 0` → fully **on-policy** (lockstep gen/train);
  `> 0` → bounded **off-policy** async (rollouts may lag the trainer by N steps).
- **Weight sync overlaps** the next training step (`WeightSyncManager`:
  push→pull→buffer-release fired in the background, ~1e-7 of step time in traces).
- Separate GPU meshes on one host (e.g. GPUs 0-3 trainer, 4-7 generator) via a
  `PerHostProvisioner` setting `CUDA_VISIBLE_DEVICES`; multi-host via Monarch
  `HostMesh`.

### 3.3 Slime control flow (from `train.py`)

- Ray creates placement groups → a **rollout manager** (SGLang engines inside) and
  **training models** (Megatron actors).
- Loop per rollout id: `rollout_manager.generate` → put in **Data Buffer** →
  `actor_model.async_train` → `actor_model.update_weights` (push to SGLang).
- Optional **critic** (PPO), **offload** of rollout/train to CPU between phases,
  periodic eval/save, `fully_async_rollout` for long-tail agentic generation.
- Customization seams: `--rollout-function-path`, `--custom-generate-function-path`,
  reward via `rm_hub`, dynamic-sampling filters. Agentic examples (multi-agent,
  search-r1, coding-agent) all plug into this one loop.

### 3.4 Miles deltas on top of Slime

- Same Ray/Megatron/SGLang skeleton, but:
  - **`miles/router/router.py`** — a bespoke async FastAPI router doing
    active-request load balancing, health checks, and dead-worker quarantine
    (beyond stock sglang-router).
  - **`miles/true_on_policy/`** — typed contracts + model profiles describing
    exactly which parallel layouts (FSDP / FSDP+TP) preserve on-policy parity.
  - **`miles/backends/experimental/fsdp_utils/`** — a full FSDP training backend
    (custom fused-MoE Triton kernels, qwen3-moe models) so Miles is not
    Megatron-only for training.
  - **`update_weight_from_distributed/{broadcast,delta}`**, FP8/MXFP8/NVFP4
    quantizer processors, `local_weight_checksum`, LoRA helpers, fault-tolerance
    (`ft/`, `nvidia-resiliency-ext`, torchft).

---

## 4. Feature Matrix

| Feature | **TitanRL** | **Slime** | **Miles** |
|---|---|---|---|
| Train backend | torchtitan (FSDP2/TP/PP/CP/EP) | Megatron | Megatron + **FSDP (experimental)** |
| Inference backend | vLLM | SGLang | SGLang (deep) |
| Orchestrator | Monarch | Ray | Ray |
| GRPO / DAPO | ✅ GRPO, DAPO | ✅ GRPO, PPO, + | ✅ GRPO, DAPO, DrGRPO, PPO |
| SFT / distillation | ⚠️ (RL focus) | ✅ SFT, on-policy distillation | ✅ + speculative online-SFT draft |
| Multi-turn / tool use | ✅ via Search-R1 rollouter | ✅ (search-r1, retool, tau-bench) | ✅ + retool-v2, SWE-agent, VLM multi-turn |
| Multi-agent | ❌ (not yet) | ✅ (`multi_agent`) | ✅ + MrlX co-evolution |
| VLM (vision) | ⚠️ (LLM-focused; notes VLM caveats) | ✅ (geo3k_vlm) | ✅ unified VLM/LLM multi-turn |
| **Bitwise train/infer parity** | ✅ **native** (unified model + batch-invariant) | ⚠️ engineered (reproducibility docs) | ✅ engineered (FP8-unify, R3, TIS/MIS) |
| FP8 rollout+train | ❌ | ⚠️ (low_precision scripts) | ✅ **Unified FP8** (headline) |
| INT4 QAT | ❌ | ❌ | ✅ (1TB model on single H200) |
| MoE routing-replay (R3) | ❌ (listed as a known gap) | ⚠️ `routing_replay.py` util | ✅ **R3** (paper + docs) |
| Speculative decoding in RL | ❌ | ❌ | ✅ (online-SFT draft, ~25% speedup) |
| LoRA | ❌ | ❌ | ✅ |
| Async / off-policy | ✅ bounded `max_offpolicy_steps` | ✅ `fully_async` | ✅ fully-async + partial rollout + over-sample |
| Weight-sync transports | RDMA (TorchStore) | NCCL / disk / disk-delta / tensor | + zero-copy CUDA-IPC, p2p, bucketed |
| Fault tolerance | ⚠️ graceful shutdown | ✅ docs | ✅ (torchft, resiliency-ext, in-mem ckpt) |
| Model coverage | Qwen3, Qwen3-MoE, GPT-OSS | GLM, Qwen, DeepSeek, Llama | GLM, Qwen, DeepSeek(V3.2), Llama4, Gemma, MiniMax, gpt-oss, … |
| Hardware | H100/H200 (FA3), A100 (FA2) | NVIDIA + **AMD** (day-0) | NVIDIA + **AMD** + **NPU** patches |
| Docker images | ❌ (uv/pip from README) | ✅ (docker/) | ✅ `radixark/miles:latest` + amd/npu patches |

Legend: ✅ first-class · ⚠️ partial/indirect · ❌ absent.

---

## 5. Cross-Functional (XFN) Footprint

*"XFN" = which teams/disciplines must be involved to build, ship, or operate the
framework, and which external projects it structurally depends on.*

### TitanRL
- **Internal (Meta/PyTorch) XFN**: torchtitan core team (model defs, FSDP/parallel),
  **Monarch** team (actor runtime), **TorchStore** team (RDMA weight transfer),
  vLLM (external but PyTorch-adjacent), and the **thinking-machines batch_invariant
  ops** dependency. Tight coupling to *bleeding-edge PyTorch nightly + vLLM
  nightly (cu130)*.
- **Discipline mix**: heavy on **numerics / kernels / distributed-systems**
  correctness; lighter on data/agentic tooling. The whole value prop (bitwise
  parity) is an infra/numerics concern.
- **Consumer**: PyTorch-ecosystem researchers who want an authoritative,
  hackable RL loop that "just matches" between train and infer.

### Slime
- **XFN dependencies**: **SGLang** team (inference), **Megatron-LM** (NVIDIA), the
  **mbridge / Pai-Megatron-Patch / veRL / OpenRLHF** lineage, Ray. Reward/verifier
  and environment authors plug in via data-generation interfaces.
- **Discipline mix**: balanced — training-systems + inference-serving + a large
  **data-generation / RL-workflow** surface (agents, sandboxes, verifiers). CI,
  reproducibility, fault-tolerance, tracing, profiling are first-class.
- **Consumer**: frontier-model labs (GLM), and a broad OSS ecosystem building
  downstream frameworks on top.

### Miles
- **XFN dependencies**: everything Slime has, **plus** an explicit enterprise
  systems surface — **NVIDIA Transformer Engine** (FP8/TE), DeepGEMM, FlashAttn-3,
  torchft / nvidia-resiliency-ext (fault tolerance), AMD & NPU vendor patches, and
  a commercial backer (RadixArk) + partners (InfiXAI, Ant Group AQ, SGLang RL
  team). Co-development *with* SGLang upstream (R3 was a joint effort).
- **Discipline mix**: the most **systems/infra-heavy** of the three — low-precision
  kernels, quantization, weight-sync transport engineering, deployment (Docker,
  multi-vendor hardware), operational tooling.
- **Consumer**: enterprises running large-scale/MoE RL in production who need
  stability, low-precision cost savings, and vendor-hardware breadth.

**XFN one-liner:** TitanRL's XFN is *inward* (Meta infra teams + numerics);
Slime's is *balanced* (train + serve + data/agents, community-driven); Miles' is
*outward + enterprise* (vendor hardware, quantization, deployment, commercial
partners).

---

## 6. Target Audience

| Audience axis | **TitanRL** | **Slime** | **Miles** |
|---|---|---|---|
| Primary user | PyTorch researchers / infra engineers | Frontier-lab RL researchers + OSS builders | Enterprise ML platform / infra teams |
| Best when you want… | A correct, hackable reference loop in the PyTorch stack | A proven, extensible substrate to run/customize real RL | Production stability + low-precision cost savings at large scale |
| Scale sweet spot | 1 node (8×H100) → small multi-node; ≤~30B MoE demonstrated | Single node → large clusters; MoE to ~355B | Very large / MoE / 1TB-class with INT4/FP8 |
| Learning curve | Moderate (small codebase, but nightly-dep + Monarch/TorchStore) | Moderate (arg pass-through keeps Megatron/SGLang native) | Higher (largest surface, most knobs) |
| "Buy" reason | Trust the numbers; extend cleanly in PyTorch | Battle-tested; freedom in data generation | Enterprise features out of the box (FP8/INT4/LoRA/FT) |
| Not for you if… | You need max-scale production throughput today | You want a non-SGLang inference backend (see vime for vLLM) | You want a minimal/simple research loop |

---

## 7. Performance & Correctness

> Direct apples-to-apples throughput numbers across all three would require
> identical hardware/model/task runs, which are **not** performed here (no GPU on
> the requesting machine; this is a theory + code comparison). What follows is
> (a) TitanRL's measured numbers from the prior 8×H100 evaluation on devvm2888,
> and (b) each project's *stated* performance posture from code/docs/blogs.

### 7.1 TitanRL — measured (Qwen3-8B, Search-R1, 8×H100, TP2/TP2, on-policy)
Batch-invariant (BI) mode ON vs OFF, per-step medians (from `rl_eval/`):

| metric (median) | BI OFF | BI ON | effect |
|---|---|---|---|
| wall-clock / step | 136.4 s | 133.6 s | ~1.0× (no net cost here) |
| generator ITL (per-token) | 9.6 ms | 22.8 ms | 2.4× slower |
| generator decode time | 177 ms | 460 ms | 2.6× slower |
| trainer fwd/bwd throughput | 370 tok/s | 127 tok/s | 2.9× slower |
| trainer full-step throughput | 11.1 tok/s | 9.2 tok/s | 1.2× slower |
| kl_div max / logprob_diff max | 0.0084 / 1.56 | **0 / 0** | **bitwise parity** |

**Interpretation:** BI's deterministic kernels make raw *compute* ~2.5–2.9×
slower, but for a rollout/orchestration-bound task (Search-R1) end-to-end
wall-clock is essentially unchanged — the compute slowdown is masked by CPU
retrieval + multi-turn orchestration + straggler sync. On a compute-bound task
the ~2.5× would surface. The payoff is *exact* generator/trainer agreement
(logprob diff = 0 every step).

**TitanRL learning-curve evidence (Search-R1, validation EM):**
- Qwen3-1.7B: EM ~0.05 → ~0.41
- Qwen3-8B: EM ~0.26 → ~0.45

### 7.2 Slime — stated posture
- Validated by **full GLM-4.5→5.2 post-training** — the strongest possible
  "it scales and converges" signal. Verified on MoE up to ~355B.
- Perf philosophy: stay **native** to Megatron + SGLang so upstream perf
  improvements (SGLang caching, PD-disaggregation, partial rollout, router
  affinity) are directly usable; avoid an abstraction tax. Partial rollout &
  fully-async address long-tail generation.

### 7.3 Miles — stated posture (systems-optimization headline numbers)
- **Zero-copy weight sync**: ~**50%** sync-time reduction vs standard HTTP/RPC
  (CUDA-IPC zero-copy mapping, async gather, bucketed flattening).
- **Speculative RL**: **25%+ rollout speedup** via an online-SFT draft model
  (draft updated during RL to avoid policy drift).
- **INT4 QAT**: fits **1TB-scale** models into single-machine VRAM (H200),
  ~doubling rollout efficiency by eliminating cross-node bottlenecks while
  claiming BF16-equivalent accuracy.
- **Unified FP8 & R3**: aimed at *correctness at scale* — eliminate the
  quantization-induced and MoE-routing-induced train/inference mismatch that
  causes RL collapse in large MoE (Qwen3, DeepSeek-V3).

### 7.4 Correctness philosophies compared (the deepest technical contrast)

| | **TitanRL** | **Slime** | **Miles** |
|---|---|---|---|
| Root approach | **Avoid** mismatch by construction (one model def) | Manage mismatch operationally (repro docs, debug replay) | **Attack** mismatch with systems + algorithms |
| Key mechanisms | Unified model in vLLM; batch-invariant Triton ops; forced NCCL tree; num_splits=1; fp32 freqs_cis | reproducibility guide, debug rollout-then-train replay, seqlen balancing | Unified FP8 quantization; **R3** routing replay; **TIS/MIS** off-policy correction; bitwise kernels (FA3/DeepGEMM) |
| Guarantee | Bitwise logprob parity *when trainer TP == generator TP* | Best-effort determinism + tooling | Bit-wise expert alignment (R3); mitigation when parity impossible (TIS/MIS) |
| Known limit | Parity only under matched parallelism; no SP; no full torch.compile in BI mode | two-stack drift must be watched | "Zero mismatch for MoE RL" still in-progress |

---

## 8. Known Gaps / Limitations (from source + docs)

### TitanRL (explicit TODOs in code / eval plan)
- Bitwise parity **requires symmetric parallelism** (trainer TP == generator TP);
  sequence parallelism unsupported in BI mode (non-deterministic reduce-scatter).
- No full `torch.compile` in batch-invariant mode.
- **No Rollout Routing Replay** yet → MoE router-mismatch untested/uncorrected.
- Weight-sync can OOM-spike on large models (GPU-Direct default; CPU staging is a
  TODO).
- Base loop is **single-turn**; multi-turn only via the Search-R1 rollouter.
- Depends on **nightly** torch + vLLM (cu130) + Monarch/TorchStore — fragile,
  fast-moving install surface.
- Experimental status; "APIs and configurations may change."

### Slime
- **SGLang-only** rollout backend by design (want vLLM? use the `vime` fork).
- Megatron-only training (no native FSDP path).
- Two-stack model design → train/infer parity is a managed concern, not a
  guarantee.
- Known operational sharp edges (e.g. Megatron+SGLang hang-after-cudagraph issues
  reported upstream).

### Miles
- Largest surface = most complexity/knobs; steeper operational learning curve.
- "**Zero mismatch for MoE RL**" and "aligning SGLang with Megatron in MoE models"
  are still **in-progress** roadmap items.
- FSDP backend is **experimental**; heavy reliance on bleeding-edge
  quantization/TE stacks and vendor patches.
- Younger project / smaller community than Slime (though co-evolving).

---

## 9. When to choose which

- **Choose TitanRL** if you live in the **PyTorch/torchtitan** world, need a
  *correctness-first* reference RL loop, care about **bitwise train/infer parity**
  and determinism for debugging, and are doing research at single-node / modest
  multi-node scale. It is the best *pedagogical + correctness* substrate, not the
  max-throughput production engine.

- **Choose Slime** if you want a **proven, general, extensible** RL substrate that
  has actually trained frontier models (GLM), want **maximum data-generation
  freedom** (agents, tools, sandboxes, verifiers) without forking the trainer, and
  are comfortable standardizing on **SGLang + Megatron**.

- **Choose Miles** if you need **enterprise/production** large-scale or MoE RL with
  **low-precision** cost savings (FP8/INT4), **routing-replay stability (R3)**,
  **speculative** rollout speedups, LoRA, fault tolerance, and **multi-vendor
  hardware** (NVIDIA/AMD/NPU) — and can absorb the extra operational complexity.

---

## 10. At-a-glance summary

| Dimension | **TitanRL** | **Slime** | **Miles** |
|---|---|---|---|
| One-word essence | **Correctness** | **Substrate** | **Enterprise** |
| Ecosystem | PyTorch / vLLM / Monarch | SGLang / Megatron / Ray | SGLang / Megatron(+FSDP) / Ray |
| Origin | Meta (PyTorch) | THUDM / Z.ai (GLM) | RadixArk (fork of Slime) |
| License | BSD-3 | Apache-2.0 | Apache-2.0 |
| Size (core LOC) | ~12K | ~29K | ~46K |
| Maturity | Experimental | Battle-tested (frontier) | Enterprise (young) |
| Superpower | Unified model → bitwise parity | Proven at frontier scale + data freedom | FP8/INT4 + R3 + spec-decode systems layer |
| Biggest limit | Nightly-dep, matched-TP parity only, single-turn base | SGLang-only, Megatron-only | Complexity; MoE-zero-mismatch WIP |

---

### Appendix A — Sources read
- **TitanRL**: `torchtitan/experiments/rl/{README.md, train.py, controller.py,
  types.py, losses/{grpo,dapo}.py, components/weight_sync.py, docs/bitwise_parity.md,
  examples/search_r1/README.md, examples/*/config_registry.py, models/vllm_registry.py}`;
  prior evaluation notes in `rl_eval/TitanRL_Eval_Results.md`.
- **Slime**: `README.md`, `train.py`, `train_async.py`, `pyproject.toml`,
  `requirements.txt`, full `slime/` + `slime_plugins/` module trees; lmsys.org
  vision blog; thudm.github.io/slime docs.
- **Miles**: `README.md`, `pyproject.toml`, `requirements.txt`, `.gitmodules`,
  full `miles/` module tree incl. `router/`, `true_on_policy/`,
  `backends/experimental/fsdp_utils/`; `docs/{advanced,user-guide}`; RadixArk /
  lmsys blogs (FP8, INT4-QAT, R3, launch announcement).
- External: GitHub, lmsys.org, PyTorch Foundation / RadixArk announcements.

*No GPU runs were executed for this comparison. TitanRL throughput/parity figures
are from a prior independent 8×H100 evaluation on devvm2888.*
