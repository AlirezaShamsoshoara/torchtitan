# PR #3819 vs. TitanRL Google Doc — Comparison & Multi-Node Run Setup

**Author of both artifacts:** Yichuan Wang (`yichuan@meta.com`)
**PR:** [pytorch/torchtitan#3819 — "[RL] update big run config"](https://github.com/pytorch/torchtitan/pull/3819) (branch `yichuan/rl-weight-sync-keep-fp32-buffers`, base `main`, state: OPEN)
**Doc:** *Qwen3-30B-A3B Async DeepEP Search-R1 for titanRL H1: Big-Run Recipe and Findings*
**W&B run:** `c912idt7` (job "v32b"), meta.wandb.io — captured ~step 193 on 2026-06-29
**This analysis generated on:** local Mac checkout, upstream fetched at `upstream/main` = `1c40dd26a`

---

## 1. TL;DR — Are they similar?

**Yes — they describe the same run and are highly consistent.** The Google doc is the
**narrative / write-up** (the "why", the debugging story, the results, the topology, and the
reproduction path). PR #3819 is the **code that bakes that recipe into the OSS repo** (the "what",
as concrete config values + three supporting correctness/observability fixes).

Every quantitative knob the doc cites matches the PR diff exactly. There are **no contradictions**
in the config values. The differences are entirely about **scope**, not disagreement:

| Aspect | Google Doc | PR #3819 |
|---|---|---|
| Purpose | Findings write-up + reproduction guide | Code change (recipe + 3 fixes) |
| Reward result | 0.25 → **0.47** (still climbing at step ~180) | Body cites 0.25 → **0.39** at step 20 (a snapshot; the doc has the later, higher number) |
| MAST wrapper details | Documented (fbcode-only, off-repo) | **Not in the PR** — lives in fbcode, not OSS |
| `off_policy_window=3` | Documented as an fbcode-wrapper override | **Not in the PR** (no such field in OSS recipe) |
| Multi-node launch (`run.sh`, retriever staging, conda env) | Documented | **Not in the PR** — fbcode submit path |
| Code fixes | Summarized in "Supporting code changes" | The actual diff for all three |

**One numeric nuance:** the PR **body** was written earlier and quotes the reward as
`0.25 (step 0) → 0.39 (step 20)` with "0 zero-gradient steps". The **doc** has the fuller,
later curve (0.25 → 0.47, new high at step 180). Same run, doc just reflects more steps. Not a
contradiction — the PR body simply hasn't been updated to the latest snapshot.

---

## 2. What PR #3819 actually changes (verified against the local diff)

Base = `51c197c86` (merge-base with the PR head). Three files, +79 / −21:

### 2a. `examples/search_r1/config_registry.py` (+55 / −16) — the recipe
Fills in the previously-`TODO: TBD` knobs of `rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf()`:

| Knob | Before (PR base) | After (PR #3819) | Matches doc? |
|---|---|---|---|
| `num_generators` | `2` (TODO) | **`4`** | ✅ doc §Scale (line 187) |
| `num_groups_per_train_step` | `32` (TODO) | **`8`** | ✅ doc §recipe (line 196) |
| `group_size` | `8` (TODO) | **`8`** | ✅ doc §recipe (line 197) |
| `drop_zero_std_reward_groups` | *(absent)* | **`True`** (via `TrainingSampleBuilder.Config`) | ✅ doc §recipe (198-200) — "THE learning fix" |
| `validation` | `num_samples=500` | `num_samples=500, **interval=20**` | ✅ doc §recipe (line 204) |
| `local_batch_size` | `1` (TODO) | **`2`** | ✅ doc §recipe (line 213) |
| `seq_len` | `4096` | `4096` | ✅ |
| trainer FSDP (`data_parallel_shard_degree`) | `8` (TODO) | **`16`** | ✅ doc §recipe (231-235) |
| trainer `tensor_parallel_degree` | `1` | `1` | ✅ |
| trainer EP (`expert_parallel_degree`) | `8` | **`16`** | ✅ doc §recipe (231-235) |
| trainer `loss` | bare `DAPOLoss(0.2, 0.28)` | **`ChunkedLossWrapper(num_chunks=16, DAPOLoss(0.2,0.28))`** | ✅ doc §recipe (246-249) |
| generator `cudagraph.mode` | `FULL_AND_PIECEWISE` | **`FULL_DECODE_ONLY`** | ✅ doc §recipe (line 263) — the decisive gen fix |
| generator override imports | `[*perf_imports, deepep_inference]` (incl. `fused_swiglu`) | **`[helion_rope, deepep_inference]`** (drops `fused_swiglu`) | ✅ doc §recipe (270-275) + narrative (f) |
| generator TP/EP/DP | `4 / 4 / 1` | `4 / 4 / 1` (unchanged) | ✅ doc §Scale |
| `max_num_batched_tokens` | `2048` | `2048` (unchanged) | ✅ doc §recipe |
| `num_max_tokens_per_rank` | `2048 // 4 = 512` | same | ✅ doc §recipe |
| `compile` | `enable=False` | `enable=False` (unchanged) | ✅ doc §recipe (line 216) |

Also adds an import of `TrainingSampleBuilder` and expands the docstring (FULL_DECODE_ONLY note).

### 2b. `controller.py` (+19 / −3) — periodic held-out validation
Adds `ValidationConfig.interval: int = 0` and, in the main training loop, a periodic validation
pass every `interval` steps (finishes the in-flight weight sync first so generators hold the
current policy; guards against double-counting the final step). Matches doc "Supporting code
changes → Periodic held-out validation."

### 2c. `observability/metrics/processor.py` (+5 / −2) — console key fix
Changes the validation console allow-list from `validation/reward/_mean|_max` to
**`validation_reward/_mean|_max`** (underscore), because `compute_rollout_metrics` emits the
reward under `<prefix>_reward`. Without the exact spelling the key is silently filtered out.
Matches doc "Supporting code changes → validation_reward console-key fix."

> **The doc also describes a 4th fix — "Weight sync keeps fp32 buffers" in `actors/trainer.py`**
> (narrative (g)). The **PR body mentions it**, but it is **NOT in this PR's diff** (the branch
> is named after it — `yichuan/rl-weight-sync-keep-fp32-buffers` — and it builds on #3825).
> It likely landed in a separate/earlier diff. So: doc and PR body both reference it; the *code*
> for it is not part of the #3819 3-file diff.

---

## 3. Multi-node cluster setup — every detail I could extract

> ⚠️ **Critical scoping note:** The **multi-node run was on MAST**, and the MAST launch machinery
> (the fbcode wrapper, `run.sh` submit path, conda env, launcher image, retriever staging, and the
> `off_policy_window=3` override) lives in **fbcode — it is NOT in the OSS repo and NOT in PR #3819.**
> The doc documents it; the PR only carries the OSS recipe values. Everything below is transcribed
> from the doc (the authoritative source for the run setup), with OSS-vs-fbcode clearly labeled.

### 3.1 Cluster topology — 5 grandteton H100 hosts total

| Role | Parallelism | GPUs | Hosts |
|---|---|---|---|
| **Controller** (+ retriever) | — | 1 host (retriever uses its last 2 GPUs) | 1 |
| **Trainer** | FSDP=16 / EP=16 / TP=1 | 16 | 2 (8 GPUs/host) |
| **Generator** (each of 4) | TP=4 / EP=4 / DP=1 | 4 each | packed 8 GPUs/host → 2 hosts |

- **Total = 1 controller + 2 trainer + 2 generator = 5 grandteton H100 hosts** (8 GPUs/host).
- Trainer: FSDP=16 × TP=1 × EP=16 → 16 GPUs = 2 hosts.
- Generators: 4 generators × 4 GPUs = 16 GPUs = 2 hosts (packed at 8/host).
- Generator TP is **pinned at 4** because Qwen3-30B-A3B has **4 KV heads** (TP must be ≤ 4).
- DeepEP v2 can span nodes (NVLink intra-node + IB/RoCE inter-node), but **each generator here is
  kept single-node** (TP=4/EP=4 fits in one host).

### 3.2 The retriever (single point / current bottleneck)
- A **single faiss-GPU server** co-located on the **controller host**, using its **last 2 GPUs**.
- Embedding model: **e5** (`intfloat/e5-base-v2`).
- Index: faiss **Flat** (`e5_Flat.index`, ~64.5 GB, fp16-sharded across 2 GPUs).
- Corpus: **wiki-18** (`wiki-18.jsonl`, ~21M passages / ~14 GB).
- `topk = 3`, served on `http://127.0.0.1:8000/retrieve`.
- Uses a cuDNN workaround to dodge a fused-MHA failure on MAST grandteton H100.
- **This single retriever is the throughput ceiling** — it saturates and starves generators
  waiting on retrieval during multi-turn rollouts.

### 3.3 The RL loop / async orchestration
- torchtitan async actor loop: **1 controller** orchestrates **trainers** (policy under FSDP/EP)
  and **generators** (vLLM engines), with weight sync push/pull between them.
- **Off-policy window = 3** (set by the **fbcode MAST wrapper**, NOT the OSS recipe): rollouts may
  be up to 3 policy-version syncs stale so generators stay busy.
  - In OSS `AsyncLoopConfig` the closest field is `target_offpolicy_steps` (default 3) +
    `window_fraction` (default 0.3) — see `docs/windowed_fifo.md`. The doc's "off_policy_window=3"
    is the fbcode wrapper's expression of this.
- Rollouts per train step: `num_groups_per_train_step (8)` × `group_size (8)` = **64 rollouts/step**.

### 3.4 fbcode MAST wrapper overrides (applied on top of the OSS recipe)
The wrapper (function referenced as `…_perf_v32b`-style in `config_registry.py` **in fbcode**,
line ref `config_registry.py:208` in the doc) starts from the OSS
`rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf()` and overrides **only**:

| Knob | Value | Source |
|---|---|---|
| `hf_assets_path` | MAST **Manifold mirror** of `Qwen3-30B-A3B` (not the OSS local path) | fbcode wrapper |
| `off_policy_window` | **3** | fbcode `config_registry.py:208` |

- The wrapper does **NOT** re-set `group_size` — the "16" in its (stale) docstring is not used;
  the run inherits `group_size=8` from the OSS recipe.
- Everything else comes from the OSS recipe (transcribed in §2a).

### 3.5 MAST submit path (fbcode)
- **Submit:** an fbcode entry point + `run.sh`. Only the **controller** role runs the retriever
  bring-up (guarded by a role check): it stages the train/test parquet, launches the retriever on
  the controller's last 2 GPUs, health-checks the retrieve endpoint until ready, then execs the
  controller with the dataset data paths appended.
- **Conda env:** a dedicated env holding **faiss-gpu (sm_90), e5, datasets, fastapi**. `run.sh`
  activates it and branches production-vs-OSS on an env flag. Platform select: aarch64/GB200 branch
  vs. the H100 branch.
- **Launcher image:** the specific MAST launcher image string lives in the fbcode submit config;
  the retriever + stack run in that conda env.
- **Observability:** W&B mirror on meta.wandb.io, run id `c912idt7` (job "v32b"); held-out
  validation reward logged every 20 steps.

### 3.6 Mixed precision (both trainer & generator)
- OSS default: **fp32 master weights + bf16 compute (DeepEP requires bf16 compute) + fp32 reduce.**
- The 30B recipe passes no explicit `mixed_precision` arg → values come from
  `TrainingConfig`/FSDP defaults; the only in-function dtype is the generator's `model_dtype="bfloat16"`.

### 3.7 Non-MAST / OSS single-machine reproduction (from `experiments/rl/README.md`)
The OSS repo does **not** ship a multi-node launcher — multi-node = MAST (fbcode). For a local /
single-node repro of the *framework* (smaller models), the OSS path is:

1. **Env:** `uv venv --python 3.12 titan-rl && source titan-rl/bin/activate`.
2. **Install:** `torchmonarch`, `torchstore` (main), `pygtrie portpicker`, `renderers` (main),
   Flash-Attention 3, PyTorch/torchvision/vllm/torchcomms nightlies (cu130), optionally
   `batch_invariant_ops`.
3. **PYTHONPATH:** `export PYTHONPATH="$PWD:${PYTHONPATH:-}"` from repo root (Monarch-spawned
   workers inherit it).
4. **Checkpoint:** download the Qwen3 HF assets into
   `torchtitan/experiments/rl/example_checkpoint`.
5. **Retriever (Search-R1):** start the dense retriever **before** training, pinned to spare GPUs:
   ```bash
   python <search-r1>/local_dense_retriever/retrieval_server.py \
     --index_path $INDEX_PATH/e5_Flat.index \
     --corpus_path $CORPUS_PATH/wiki-18.jsonl \
     --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
   ```
6. **Run:** `python -m torchtitan.experiments.rl.train --module search_r1 --config <recipe>`
   (the big config is `rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf`, but it needs the MAST-scale
   cluster — 5 hosts — to actually run; use the 1.7B/8B recipes for a laptop/single-node smoke test).
- Orchestration across GPU meshes is done by **Monarch** (the controller framework); it spawns the
  trainer and generator proc-meshes on disjoint GPUs.

---

## 4. The debugging findings (doc narrative ↔ PR knobs)

The doc's 8 findings map 1:1 onto the PR's knobs / fixes:

| # | Finding | Fix in PR |
|---|---|---|
| (a) | DeepEP cudagraph corruption (garbage, 0 reward, ~505-tok) | `cudagraph.mode = FULL_DECODE_ONLY` (was FULL_AND_PIECEWISE) |
| (b) | Flat reward, ~33% zero-grad steps — **THE learning fix** | `drop_zero_std_reward_groups = True` |
| (c) | Trainer OOM on full-vocab (151936) × seq loss | `ChunkedLossWrapper(num_chunks=16)` |
| (d) | FSDP8/EP8 "unspecified launch failure" at 80GB edge | FSDP=16 / EP=16 (2 hosts) |
| (e) | `local_batch_size=1` → too many inter-node EP all-to-all rounds | `local_batch_size = 2` |
| (f) | `fused_swiglu` nested-override conflict (#3778) with `deepep_inference` | generator drops `fused_swiglu`, keeps `helion_rope`+`deepep_inference` |
| (g) | Weight-sync bf16 cast wrongly hit fp32 buffers (+ FQN normalization) | `actors/trainer.py` — **referenced by PR body, not in #3819's 3-file diff** |
| (h) | FSDP32/EP8 over 4 nodes = **no speedup** (negative result) | kept FSDP16/EP16 on 2 nodes (documented in the parallelism comment) |

---

## 5. Results (from the doc)

**Held-out validation reward** (pure exact-match, 500 samples):

| Step | Reward |
|---|---|
| 0 | 0.25 (cold-start) |
| 20 | 0.39 |
| 40 | 0.43 |
| 60 | 0.43 |
| 80 | 0.43 |
| 100 | 0.39 |
| 120 | 0.42 |
| 140 | 0.39 |
| 160 | 0.46 |
| 180 | **0.47** (new high; run continuing) |

- Net: **0.25 → 0.47, ~+0.2 absolute.** Before `drop_zero_std=True` the curve was flat at
  ~0.20–0.25 for 100+ steps.
- **Grad health:** `grad_norm==0` in **0 of 182 steps** (was ~33% dead before the fix).
- **Generation:** validation response ~17 tokens (terse, correct — searches then answers) vs.
  the corrupt cudagraph path's ~505-token garbage that never searched.
- **Bottleneck:** generation/rollout-bound (`generate` fraction ~0.61–0.67; `train` ~0.33–0.39);
  wall clock ~80–90 s/step. The single faiss retriever is the ceiling.

---

## 6. Bottom line

- **The doc and PR #3819 are consistent and describe the same run** — the doc is the write-up,
  the PR is the OSS code. No config value disagrees.
- **The PR is an OSS-only slice.** The **multi-node cluster setup was MAST (fbcode)**: the launch
  wrapper, `run.sh`, conda env, launcher image, retriever staging, and `off_policy_window=3` are
  **not** in the OSS repo or this PR — only the recipe values + 3 correctness/observability fixes are.
- **To reproduce the big run** you need the fbcode MAST wrapper on a 5-host grandteton H100 cluster.
  OSS gives you the recipe + framework; the OSS README only documents single-machine / smaller-model
  runs via Monarch.
- Minor: the PR **body** quotes an earlier reward snapshot (0.39@step20); the **doc** has the fuller
  0.47 curve. Same run.

---

## 7. How to run this on MAST yourself — step by step

> **Source & honesty note.** The concrete fbcode paths / names below are transcribed from the
> Google doc's "How to reproduce" + "Scale and topology" sections (authoritative — written by the
> run's author, Yichuan Wang). They live in **fbcode (`fbsource`), not in this OSS repo**, so I could
> **not** independently verify them from this Mac checkout. Treat exact paths as "the doc says X —
> confirm in fbsource before relying on it." The OSS pieces (§7.0, §7.6) are verified against this repo.

### 7.0 One-time prerequisites (before anything MAST)
1. **fbsource checkout** with the RL MAST launcher present. On a devvm:
   ```bash
   # confirm the fbcode MAST launcher dir exists
   ls fbcode/<...>/fb/mast_rl/            # expect: run.sh, mast.py, main.py, config_registry.py
   ```
   (The doc references `fb/mast_rl/{run.sh,mast.py,main.py,config_registry.py}`. Find the real
   absolute path in fbsource with e.g. `fbgs "search_r1_qwen3_30b_a3b_deepep"`.)
2. **The OSS recipe must be on the branch/commit you launch from.** PR #3819 fills in the
   `rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf()` values; the MAST wrapper starts from that OSS
   recipe. Launch from a source rev that includes PR #3819 (or the equivalent internal mirror).
3. **Model assets on Manifold.** The MAST wrapper points `hf_assets_path` at the Manifold mirror
   `/mnt/torchtrain_datasets/tree/qwen3/Qwen3-30B-A3B`. Confirm it exists / you have read access.
4. **Retriever assets on Manifold** (staged by the controller at launch, but must exist):
   - faiss **Flat** index `e5_Flat.index` (~64.5 GB, fp16, sharded across 2 GPUs)
   - corpus `wiki_dump.jsonl` (~21M passages / ~14 GB)
   - embedding model `intfloat/e5-base-v2` (e5)
   - train/test parquet (NQ / HotpotQA) for Search-R1.
5. **Capacity:** you need **5 grandteton H100 hosts (8 GPUs each = 40 GPUs)** reservable via MAST:
   1 controller + 2 trainer + 2 generator. (See §7.4 for why 5.)
6. **Conda env:** `rlmast` (or `rlmast_deepep`) containing **faiss-gpu (sm_90), e5, datasets,
   fastapi**. `run.sh` activates it via `CONDA_DIR`/`CONDA_PREFIX` and branches production-vs-OSS on
   `TRITON_LIBCUDA_PATH`. Platform: `platform010-aarch64` for aarch64/GB200, else `platform010`.

### 7.1 Understand what the MAST wrapper does (so you can adjust it)
The fbcode wrapper `search_r1_qwen3_30b_a3b_deepep` (in `fb/mast_rl/config_registry.py`):
- **starts from** the OSS `rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf()` recipe, and
- **overrides only two things:**
  1. `hf_assets_path` → `/mnt/torchtrain_datasets/tree/qwen3/Qwen3-30B-A3B` (Manifold mirror)
  2. `async_loop.max_offpolicy_steps = 3`  ← the off-policy staleness window
- **attaches** a `MastLauncher.Config` (host counts, image, roles).
- It does **NOT** re-set `group_size` (inherits `8` from OSS; the "16" in the stale docstring is unused).
- Everything else (FSDP16/EP16, 4 generators, FULL_DECODE_ONLY, ChunkedLoss, etc.) comes from OSS.

> To change scale, edit the `MastLauncher.Config` host counts + the OSS parallelism degrees together
> (they must stay consistent: trainer GPUs = FSDP×TP×EP with EP≤GPUs; generator GPUs = num_generators×TP×EP).

### 7.2 Submit the run
The doc's submit path:
```bash
# in fbsource, from the fb/mast_rl launcher dir
cd fbcode/<...>/fb/mast_rl

# run.sh activates the conda env and dispatches roles; the launcher (mast.py) reserves the
# MAST hosts, sets the image, and starts controller/trainer/generator roles.
./run.sh <args-for search_r1_qwen3_30b_a3b_deepep>
```
- **Only the controller role** runs `python mast_rl/main.py`. It is guarded by a `MANIFUSE_BUCKET`
  check, and on the controller it:
  1. stages the train/test parquet,
  2. launches the faiss-GPU retriever on `127.0.0.1:8000` (using the controller host's **last 2 GPUs**),
  3. **health-checks `POST /retrieve` until ready**, then
  4. execs the controller process with the dataset data paths appended.
- The **MAST launcher image string lives in `mast.py`** — adjust it there if you need a different image.
- Trainer + generator roles are started by the launcher on their own hosts and connect back to the
  controller (Monarch orchestrates the trainer/generator proc-meshes across the reserved hosts).

> Exact `run.sh` flags aren't spelled out in the doc — inspect `run.sh`/`mast.py`/`main.py` in
> fbsource for the actual argument names and the config selector for `search_r1_qwen3_30b_a3b_deepep`.

### 7.3 The retriever (single point — expect it to be your ceiling)
The controller brings this up for you, but know its shape so you can debug / scale it:
- faiss **Flat** index `e5_Flat.index` (~64.5 GB, fp16, across 2 GPUs), corpus `wiki_dump.jsonl`
  (~21M passages), embedding model `intfloat/e5-base-v2`, `topk=3`, served on
  `http://127.0.0.1:8000/retrieve`.
- Uses `attn_implementation="eager"` to dodge a cuDNN fused-MHA failure on MAST grandteton H100.
- **This single retriever saturates and starves generators** (the run is rollout-bound). If you want
  more throughput, replicate/shard the retriever — the doc's #1 "next lever".

### 7.4 The exact cluster to reserve (must match the parallelism)
| Role | Parallelism | GPUs | Hosts (8 GPU/host) |
|---|---|---|---|
| Controller (+ retriever) | — | uses last **2 GPUs** for faiss | **1** |
| Trainer | FSDP=16 / TP=1 / EP=16 | 16 | **2** |
| Generator ×4 | each DP=1 / TP=4 / EP=4 | 4 each = 16 | **2** (packed 8/host) |
| **Total** | | **40 GPUs** | **5 hosts** |

- Generator **TP is pinned at 4** (Qwen3-30B-A3B has 4 KV heads → TP ≤ 4). Don't raise it.
- **Do NOT go to FSDP=32/EP=8 over 4 trainer nodes** — the doc's finding (h): it gave **no speedup**
  (FSDP all-gather goes inter-node and cancels the gain). FSDP=16/EP=16 on 2 nodes is the sweet spot.
- FSDP=8/EP=8 on **one** trainer host **OOMs** (finding (d)) — 2 trainer hosts is the floor.

### 7.5 Observe / verify it's actually learning
- **W&B:** meta.wandb.io, run id pattern `torchtitan-rl-search_r1_qwen3_30b_a3b_deepep-<id>`
  (the reference run was `...-d14084`, job "v32b").
- **Watch `validation_reward/_mean`** — logged every `ValidationConfig.interval=20` steps. Healthy =
  it climbs off cold-start ~0.25 (reference run reached 0.47 by step ~180).
- **Sanity checks that the fixes are active:**
  - `grad_norm==0` should be **~0%** of steps (proves `drop_zero_std_reward_groups=True` is working).
    If ~33% of steps are dead, the drop-zero-std flag isn't on.
  - `validation response_length/mean` ≈ **17 tokens** (terse, correct). If you see ~505-token garbage
    that never searches, the generator cudagraph is wrong (should be `FULL_DECODE_ONLY`, not
    `FULL_AND_PIECEWISE`).
  - No trainer OOM at start ⇒ `ChunkedLossWrapper(num_chunks=16)` is in place.
- Wall clock ≈ **80–90 s/step**; `step_time_ratio/batch` (generation) ~0.61–0.67 (rollout-bound).

### 7.6 Non-MAST / OSS fallback (single machine, smaller model — for a smoke test)
The OSS repo has **no** multi-node launcher; the 30B big-run genuinely needs the 5-host MAST cluster.
To exercise the *framework* locally (verified against this repo's `experiments/rl/README.md`):
```bash
# 1. env
pip install uv && uv venv --python 3.12 titan-rl && source titan-rl/bin/activate
# 2. deps
uv pip install torchmonarch
uv pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install pygtrie portpicker
uv pip install "git+https://github.com/PrimeIntellect-ai/renderers.git@main"
uv pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130
uv pip install torch torchvision vllm torchcomms --pre \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 --index-strategy unsafe-best-match
# 3. PYTHONPATH (Monarch workers inherit it)
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# 4. model assets
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B \
  --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...
# 5. retriever (Search-R1) — start BEFORE training, pinned to spare GPUs
python <search-r1>/local_dense_retriever/retrieval_server.py \
  --index_path $INDEX_PATH/e5_Flat.index --corpus_path $CORPUS_PATH/wiki-18.jsonl \
  --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
# 6. run (small recipe as a smoke test)
python -m torchtitan.experiments.rl.train --module search_r1 --config rl_grpo_qwen3_1_7b_search_r1
```
The 30B recipe name is `rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf`, but running *that* one at scale
still requires the MAST cluster in §7.4 — locally, use the 1.7B/8B recipes to validate your setup.

### 7.7 Checklist (tl;dr of what YOU need to do)
- [ ] fbsource checkout with `fb/mast_rl` launcher (find real path via `fbgs`)
- [ ] Source rev that includes PR #3819's OSS recipe
- [ ] Manifold: `Qwen3-30B-A3B` model mirror + `e5_Flat.index` + `wiki_dump.jsonl` + e5 model + parquet
- [ ] Conda env `rlmast`/`rlmast_deepep` (faiss-gpu sm_90, e5, datasets, fastapi)
- [ ] Reserve **5 grandteton H100 hosts (40 GPUs)**: 1 controller + 2 trainer + 2 generator
- [ ] (Optional) tweak `MastLauncher.Config` in the fbcode wrapper for host counts / image
- [ ] Submit via `fb/mast_rl/run.sh` selecting `search_r1_qwen3_30b_a3b_deepep`
- [ ] Confirm controller staged parquet + retriever healthy on `:8000` before trainer/gen connect
- [ ] Watch W&B `validation_reward/_mean` climb; verify `grad_norm==0` ≈ 0% and response ≈ 17 tokens
