# TitanRL on MAST — Verified Runbook for the PR #3819 Big Run

**What this is:** a step-by-step guide to running the Qwen3-30B-A3B DeepEP Search-R1 big run
(PR [#3819](https://github.com/pytorch/torchtitan/pull/3819)) on MAST, written by **reading the
actual `fb/mast_rl` launcher code** in fbsource — not just the Google doc.

**Source of truth for this file:**
- fbsource checkout on devvm2888: `/home/alisol/fbsource-navi-guac` @ rev `dd4fa114fd17` (2026-07-28)
- Launcher dir: `fbcode/pytorch/torchtitan/fb/mast_rl/` (`README.md`, `submit.sh`, `launcher.py`,
  `mast.py`, `main.py`, `run.sh`, `local.py`, `config_registry.py`, `env_defaults.py`, `build_conda.sh`)
- OSS recipe: `torchtitan/experiments/rl/examples/search_r1/config_registry.py`

> Companion file: `PR3819_vs_TitanRL_doc_analysis.md` (the doc-vs-PR comparison). Read that first
> for *what* the run is; this file is *how* to launch it, corrected against the real code.

---

## ⚠️ 0. Read this first — where the Google doc and the real launcher DIVERGE

The doc's "How to reproduce" section describes an idealized setup. Cross-checked against the actual
`fb/mast_rl` code, **several of its claims do not match what the launcher does today.** These are not
nitpicks — they change the commands you run and the hosts you reserve.

| Doc claim | Reality in `fb/mast_rl` (rev dd4fa114fd17) | Impact |
|---|---|---|
| A recipe `search_r1_qwen3_30b_a3b_deepep` exists in `fb/mast_rl/config_registry.py` | **It does NOT exist.** `config_registry.py` has only 3 recipes, all **AlphabetSort**: `test_0_6b`, `test_30b_multi_hosts`, `test_rl_grpo_qwen3_30b_a3b_varlen`. `fbgs "search_r1_qwen3_30b_a3b_deepep"` across fbsource returns **nothing**. | **You must add your own MAST recipe** wrapping the OSS Search-R1 config (§3). |
| The controller "launches the retriever on 127.0.0.1:8000, health-checks POST /retrieve until ready" | **No retriever/faiss code exists** anywhere in `mast_rl/` or the `search_r1` example. `main.py` only spawns proc meshes + runs the controller loop. `run.sh` only mounts Manifold + sets CUDA/NCCL env. | **You must run the faiss retriever yourself** (§5). The Search-R1 `env.py` just reads `search_url` (default `http://127.0.0.1:8000/retrieve`). |
| Wrapper overrides `off_policy_window=3` | The field is `AsyncLoopConfig.max_offpolicy_steps`, and its **default is already `3`** (`controller.py:166`). | Nothing to override for that value; set it only if you want ≠3. |
| Submit via `run.sh + mast.py`; controller runs `main.py` guarded by a `MANIFUSE_BUCKET` check | Submit is `submit.sh → launcher.py → mast.py`. `run.sh` is the **on-node entrypoint** (per role), `MANIFUSE_BUCKET` gates the **Manifold mount in run.sh**, not a controller code path. | Use `submit.sh` (§4), not `run.sh`, from the dev server. |
| "5 grandteton hosts: 1 controller + 2 trainer + **2 generator hosts (packed 8/host)**" | The launcher gives **each generator its own WHOLE host** (`MAST_WHOLE_HOST_FEATURE=True`, one MAST role per generator). It does **NOT pack** two generators onto a host. `num_generators=4` with generator world size 4 → **4 generator hosts** (each using 4 of 8 GPUs). | PR #3819's config → **7 hosts** (1+2+4), not 5. To get exactly 5, use `num_generators=2` (§3/§6). |

**Bottom line:** the OSS PR #3819 landed the *recipe values*; the doc's *MAST launch story is partly
aspirational*. The reusable MAST machinery that DOES exist is the generic `fb/mast_rl` launcher +
AlphabetSort test recipes. To run the **Search-R1 big run** you glue the two together yourself (§3–§5).

---

## 1. How the `fb/mast_rl` launcher actually works (verified)

```diagram
  dev server                                   MAST (grandteton_80g_roce, 8 GPU/host)
  ┌───────────────────────┐                    ┌──────────────────────────────────────┐
  │ submit.sh              │                    │ controller role (1 whole host)         │
  │  ├─ pick/activate env  │   monarch create   │   run.sh → python main.py --job-name J │
  │  ├─ pip reinstall      │ ─────────────────► │   attaches to trainer + generator      │
  │  │   torchtitan (.)    │   3-role AppDef     │   host meshes, spawns proc meshes,     │
  │  └─ launcher.py        │                    │   runs Controller.run()                │
  │      └─ mast.py:submit │                    ├──────────────────────────────────────┤
  │         infer_nodes()  │                    │ trainer role  (trainer_nodes hosts)    │
  │         build AppDef   │                    │   run.sh → Monarch simple bootstrap    │
  │         conda-pack env │                    ├──────────────────────────────────────┤
  └───────────────────────┘                    │ generator_0..N roles (1 host each)     │
                                                │   run.sh → Monarch simple bootstrap    │
                                                └──────────────────────────────────────┘
```

- **`submit.sh`** (dev server entry): selects the conda env (`--fbpkg` fetches
  `torchtitan_conda_prod`, else uses the active env), **reinstalls torchtitan from the fbsource repo
  root** (so your local `torchtitan/` edits ship — skip with `--no-reinstall`), pre-stages the HF
  example dataset into the env's HF cache, then runs `launcher.py`.
- **`launcher.py` → `mast.py:MastLauncher.submit`**: parses the config, reads its `launcher`
  (`MastLauncher.Config`) field, **infers per-role host counts from the parallelism**, builds a
  **3-role AppDef** (controller + trainer + one role per generator), conda-packs the active env +
  ships `mast_rl/`, and calls `monarch.tools.commands.create`.
- **Host-count inference** (`mast.py:infer_nodes`, `train.py`):
  - `trainer_world_size = dp_replicate × dp_shard × TP × PP × CP` (**EP is NOT a factor**).
  - `generator_world_size = DP × TP` (per generator).
  - `nodes = 1 if world_size ≤ gpus_per_node else world_size / gpus_per_node` (must divide evenly;
    multi-host roles tile **whole** hosts).
- **`run.sh`** (on every MAST node): platform010 `LD_PRELOAD`, libcuda/nvidia-ml symlinks,
  NCCL/CUDA env, deep_ep's nvshmem on `LD_LIBRARY_PATH`, **mounts the `MANIFUSE_BUCKET` Manifold
  bucket at `/mnt/<bucket>`**, mounts warm storage at `/mnt/wsfuse`, activates conda, then execs
  either `main.py` (controller) or the Monarch worker bootstrap (trainer/generators).
- **`main.py`**: with `--job-name` = detached MAST controller (attaches to trainer+generator host
  meshes via `MASTJob.from_torchx`, spawns proc meshes, runs the loop); without it = local run.
- **DeepEP/HybridEP env** (`env_defaults.py:DEEP_EP_ENV`): MNNVL kill-switches + NVSHMEM/IBGDA RoCE
  GID index 3 for grandteton H100; `mast.py` also sets `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN`
  **per role** = `min(ep_size, gpus_per_node)`.

### The 3 recipes that DO exist in `fb/mast_rl/config_registry.py`
| Recipe | Task | Trainer | Generator | num_gen | Hosts (8/host) |
|---|---|---|---|---|---|
| `test_0_6b` | AlphabetSort (dense 0.6B) | inherits 0.6B recipe | inherits | default | small |
| `test_30b_multi_hosts` | AlphabetSort (30B MoE) | FSDP=32/TP=1/EP=8 → 32 GPU → **4 hosts** | DP=2/TP=4/EP=8 → 8 GPU → 1 host each | 3 → **3 hosts** | 1+4+3 = **8** |
| `test_rl_grpo_qwen3_30b_a3b_varlen` | AlphabetSort (30B MoE, HybridEP) | FSDP=8/TP=1/EP=8 → 8 GPU → **1 host** | DP=1/TP=4/EP=4 → 4 GPU → 1 host each | 2 → **2 hosts** | 1+1+2 = **4** |

**None of these is Search-R1.** They exercise the launcher + MoE backends on the self-contained
AlphabetSort task. You will add a Search-R1 recipe next.

---

## 2. Prerequisites (one-time)

1. **fbsource checkout** with `fb/mast_rl` present (you have it on devvm2888:
   `/home/alisol/fbsource-navi-guac/fbcode/pytorch/torchtitan`).
2. **A `torchtitan_conda_prod` conda fbpkg** (the env that ships to MAST). Easiest: let `submit.sh`
   fetch `torchtitan_conda_prod:stable` via `--fbpkg`. It has torch/vLLM/torchcomms cu130, CUDA 13
   toolkit, torchstore, deep_ep (HybridEP/DeepEP), Meta monarch + torchx. (Backup: build a local
   `rlmast` env with `bash fb/mast_rl/build_conda.sh`.)
3. **Model on Manifold.** Upload the Qwen3-30B-A3B checkpoint to the bucket mounted at `/mnt`:
   ```bash
   manifold putr Qwen3-30B-A3B manifold://torchtrain_datasets/tree/qwen3/Qwen3-30B-A3B
   ```
   The launcher mounts `manifold://<manifold_bucket>` at `/mnt/<bucket>`, so set
   `hf_assets_path=/mnt/torchtrain_datasets/tree/qwen3/Qwen3-30B-A3B`.
4. **Search-R1 retriever assets** (faiss Flat index `e5_Flat.index` ~64.5GB fp16, corpus
   `wiki_dump.jsonl` ~21M passages, `intfloat/e5-base-v2`), plus the NQ/HotpotQA parquet
   (`PeterJinGo/nq_hotpotqa_train`). The retriever is **your** responsibility to run (§5).
5. **MAST capacity** on `grandteton_80g_roce` (8×H100/host) in one region (default `pci`). See §6
   for exact host count.

---

## 3. Add a Search-R1 MAST recipe (because none exists)

The launcher only runs `--module mast_rl --config <fn>` recipes. Add a function to
`fbcode/pytorch/torchtitan/fb/mast_rl/config_registry.py` that (a) imports the OSS Search-R1 30B
recipe, (b) points `hf_assets_path` at the Manifold mirror, and (c) attaches a `MastLauncher.Config`.
This mirrors exactly what `test_rl_grpo_qwen3_30b_a3b_varlen` does for AlphabetSort.

```python
# add near the other imports at the top of fb/mast_rl/config_registry.py
from torchtitan.experiments.rl.examples.search_r1.config_registry import (
    rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf,
)


def search_r1_qwen3_30b_a3b_deepep() -> MastRLConfig:
    """PR #3819 big run: Qwen3-30B-A3B DeepEP Search-R1 on MAST.

    Trainer FSDP=16/TP=1/EP=16 -> 16 GPU -> 2 hosts. Generator DP=1/TP=4/EP=4 -> 4 GPU
    -> 1 host each. num_generators picks the generator host count (see §6). The OSS
    recipe already carries every PR #3819 knob (drop_zero_std=True, FULL_DECODE_ONLY,
    ChunkedLoss num_chunks=16, local_batch_size=2, validation interval=20, etc.).
    """
    config = rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf()  # OSS recipe = PR #3819 values
    # Manifold mirror (mounted at /mnt/<bucket> on MAST):
    config.hf_assets_path = "/mnt/torchtrain_datasets/tree/qwen3/Qwen3-30B-A3B"
    # off-policy window: default is already 3; set explicitly only if you want that documented.
    config.async_loop.max_offpolicy_steps = 3
    # Point Search-R1 at your retriever (see §5). Default is http://127.0.0.1:8000/retrieve,
    # which only works if the retriever runs ON each generator host. For a shared retriever host,
    # set the reachable URL here, e.g.:
    # config.rollouter.message_env.search_url = "http://<retriever-host>:8000/retrieve"
    return _with_launcher(config, MastLauncher.Config(
        host_type="grandteton_80g_roce",   # 8×H100/host
        region="pci",
        manifold_bucket="torchtrain_datasets",
    ))
```

> ⚠️ **Verify the OSS recipe matches PR #3819 in your checkout.** In `fbsource-navi-guac` @
> `dd4fa114fd17` the OSS `rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf()` is an **older, pre-#3819**
> version (`num_generators=2`, `local_batch_size=1`, FSDP=8/EP=8, `FULL_AND_PIECEWISE`,
> `deepep_override`, no `drop_zero_std`). **PR #3819 has not landed in this fbsource checkout yet.**
> Either (a) rebase/pull fbsource to a rev that includes #3819, or (b) set the knobs explicitly in
> your MAST recipe to the #3819 values (FSDP=16/EP=16, num_generators=4, local_batch_size=2,
> group_size=8, num_groups_per_train_step=8, drop_zero_std=True, cudagraph=FULL_DECODE_ONLY,
> ChunkedLossWrapper(num_chunks=16, DAPOLoss(0.2,0.28)), validation interval=20, generator override
> = helion_rope + deepep_inference only). The exact values are in
> `PR3819_vs_TitanRL_doc_analysis.md` §2a — verified against the PR diff.

After editing, `arc f fb/mast_rl/config_registry.py` before any diff.

---

## 4. Submit to MAST

From the fbsource repo root's `fb/` context (the dev server, devvm2888):

```bash
cd /home/alisol/fbsource-navi-guac/fbcode/pytorch/torchtitan

# Option A: let submit.sh fetch the prod conda env (simplest)
bash fb/mast_rl/submit.sh \
  --fbpkg torchtitan_conda_prod:stable \
  --config search_r1_qwen3_30b_a3b_deepep \
  --launcher.region=pci \
  --launcher.host-type=grandteton_80g_roce

# Option B: run YOUR OSS torchtitan checkout instead of fbsource's copy
#   (needed if PR #3819 is only in an OSS clone, not in this fbsource rev)
source ~/fbsource/genai/msl/dev/xl_conda.sh activate torchtitan_conda_prod:stable
pip install --no-build-isolation --no-deps --force-reinstall /path/to/your/torchtitan
bash fb/mast_rl/submit.sh --no-reinstall \
  --config search_r1_qwen3_30b_a3b_deepep --launcher.region=pci
```

- `--config` is **required** and always uses `--module mast_rl`.
- Any other `--foo.bar=...` is forwarded verbatim to `ConfigManager` (overrides the recipe). E.g.
  `--num_generators 2`, `--trainer.parallelism.data_parallel_shard_degree=16`,
  `--hf_assets_path=/mnt/...`, `--launcher.gpus-per-node=4`.
- **Local dry-run first** (catches conda/asset/path issues cheaply, uses the dev server's GPUs):
  ```bash
  # mount the manifold bucket locally so /mnt/... resolves
  oilfs --profile manifold manifold://torchtrain_datasets /mnt/torchtrain_datasets
  bash fb/mast_rl/submit.sh --fbpkg torchtitan_conda_prod:stable --local \
    --config search_r1_qwen3_30b_a3b_deepep
  ```
  (Local mode needs enough free GPUs on the dev server for the chosen parallelism — the full 30B
  config won't fit one box; shrink it or use `test_0_6b` to validate the pipeline.)

The submit prints the inferred topology, e.g.:
```
inferred from config (8 GPUs/host): trainer 16 GPUs -> 2 host(s),
4 generator role(s) x 4 GPUs -> 1 host(s) each
```
Job handle is `mast_conda:///torchtitan-rl-search_r1_qwen3_30b_a3b_deepep-<uuid>`.

---

## 5. Run the faiss retriever yourself (the launcher does NOT)

Search-R1 rollouts POST to `search_url` (default `http://127.0.0.1:8000/retrieve`). **Nothing in
`fb/mast_rl` starts a retriever** — you must run it and make it reachable from the generators.

Two workable patterns:
1. **Retriever co-located on each generator host at `127.0.0.1:8000`** (matches the default URL).
   You'd need a way to start the retriever process on the generator roles — the current worker
   bootstrap doesn't do this, so this requires launcher changes (a `run.sh` branch that starts the
   retriever on generator roles) OR a sidecar.
2. **A dedicated retriever host** (what the doc's topology implies — retriever on the controller's
   last 2 GPUs). Start the retriever there and set `search_url` to that host (§3). The doc's run put
   it on the **controller** host's spare GPUs.

The retriever command (from the OSS `search_r1/README.md`):
```bash
python <search-r1>/local_dense_retriever/retrieval_server.py \
  --index_path $INDEX_PATH/e5_Flat.index \
  --corpus_path $CORPUS_PATH/wiki_dump.jsonl \
  --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
```
- Uses `attn_implementation="eager"` on grandteton H100 to dodge a cuDNN fused-MHA failure.
- It saturates as a single point — the run's throughput ceiling. Replicating it is the doc's #1
  next lever.

> **This is the biggest gap between "recipe landed" and "run reproducible."** Budget time to wire
> the retriever into your launch (a `run.sh`/`main.py` change to spawn + health-check it), or accept
> a manual retriever host and point `search_url` at it.

---

## 6. Exactly how many hosts to reserve

Host counts are **inferred from parallelism** by `mast.py`, and **each generator role takes a whole
host** (or whole hosts). Do the math for the config you submit:

**PR #3819 config as written (num_generators=4):**
| Role | Parallelism | World size | GPUs/host | Hosts |
|---|---|---|---|---|
| Controller | — | — | — | **1** |
| Trainer | FSDP=16 × TP=1 (EP not counted) | 16 | 8 | **2** |
| Generator ×4 | each DP=1 × TP=4 = 4 | 4 each | 8 | **1 each → 4** |
| **Total** | | | | **7 hosts (56 GPUs)** |

- 4 GPUs on each generator host sit **idle** (role world size 4 < 8/host). To use them fully without
  extra hosts, submit `--launcher.gpus-per-node=4` (declares 4-GPU hosts; still 1 host per generator,
  now fully used) — but that's still **4 generator hosts**.

**To match the doc's "5 hosts" you must reduce generators to 2:**
| Role | Parallelism | Hosts |
|---|---|---|
| Controller (+ retriever on spare GPUs) | — | 1 |
| Trainer | FSDP=16/TP=1/EP=16 | 2 |
| Generator ×2 | each DP=1/TP=4/EP=4 | 2 |
| **Total** | | **5 hosts (40 GPUs)** |
Submit with `--num_generators 2`. (Trade-off: fewer generators = less rollout throughput, and the
run is already rollout-bound — see the doc. The doc's own body says `num_generators=4`; its topology
table's "2 generator hosts" is inconsistent with 4 whole-host generator roles.)

**Do NOT** scale the trainer to FSDP=32/EP=8 over 4 hosts — the doc's finding (h): **no speedup**
(FSDP all-gather goes inter-node and cancels the gain). FSDP=16/EP=16 on 2 hosts is the sweet spot.
FSDP=8/EP=8 on one host **OOMs** (finding (d)).

---

## 7. Observe / verify it's learning

- **TensorBoard** (MAST forces `enable_tensorboard=True`, `enable_wandb=False`): the MAST job's TB
  tab points at the warm-storage dump dir `/mnt/wsfuse/outputs/<job_name>` (wired by
  `append_tb_logdir_metadata`). W&B is offline-only on MAST (`WANDB_MODE=offline`).
- **Watch `validation_reward/_mean`** — logged every `ValidationConfig.interval=20` steps (this is the
  console-key fix from PR #3819; without it the key is silently filtered). Healthy: climbs off
  cold-start ~0.25 (reference run reached 0.47 by step ~180).
- **Sanity checks that the #3819 fixes are active:**
  - `grad_norm==0` ≈ **0%** of steps ⇒ `drop_zero_std_reward_groups=True` working (was ~33% before).
  - validation `response_length/mean` ≈ **17 tokens** (terse, searches then answers). ~505-token
    garbage ⇒ wrong generator cudagraph (must be `FULL_DECODE_ONLY`, not `FULL_AND_PIECEWISE`).
  - no trainer OOM at start ⇒ `ChunkedLossWrapper(num_chunks=16)` in place.
- Wall clock ≈ **80–90 s/step**; `step_time_ratio/batch` (generation) ~0.61–0.67 (rollout-bound —
  the single retriever is the ceiling).

---

## 8. Step-by-step checklist (tl;dr)

- [ ] fbsource checkout with `fb/mast_rl` (have it on devvm2888)
- [ ] **Confirm PR #3819 is in your torchtitan source** (it is NOT in `fbsource-navi-guac`@dd4fa114 —
      pull a newer fbsource rev, or use an OSS clone via Option B, or set the knobs explicitly in §3)
- [ ] Upload `Qwen3-30B-A3B` to Manifold `torchtrain_datasets/.../Qwen3-30B-A3B`
- [ ] Stage retriever assets (`e5_Flat.index`, `wiki_dump.jsonl`, e5 model) + NQ/HotpotQA parquet
- [ ] **Add a `search_r1_qwen3_30b_a3b_deepep()` MAST recipe** to `fb/mast_rl/config_registry.py` (§3)
- [ ] **Solve the retriever** (§5): dedicated host + set `search_url`, or add launcher code to spawn it
- [ ] Decide host count (§6): 7 hosts for num_generators=4, or 5 hosts for num_generators=2
- [ ] `arc f fb/mast_rl/config_registry.py`
- [ ] Local dry-run: `submit.sh --fbpkg torchtitan_conda_prod:stable --local --config test_0_6b`
      (pipeline sanity) then the real config if it fits
- [ ] Submit: `bash fb/mast_rl/submit.sh --fbpkg torchtitan_conda_prod:stable --config search_r1_qwen3_30b_a3b_deepep --launcher.region=pci`
- [ ] Confirm the printed inferred topology matches your reservation
- [ ] Watch TensorBoard `validation_reward/_mean` climb; verify grad_norm==0 ≈ 0% and response ≈ 17 tokens

---

## Appendix — file-by-file reference (`fb/mast_rl/`)

| File | Role |
|---|---|
| `submit.sh` | Dev-server entry. Env selection (`--fbpkg`/active), torchtitan reinstall, HF dataset pre-stage, dispatch to `launcher.py` (MAST) or `local.py` (`--local`). Flags: `--fbpkg`, `--local`, `--no-reinstall`, `--wait` (CI). |
| `launcher.py` | Parses config, reads `launcher` field, calls `MastLauncher.submit`. |
| `mast.py` | `MastLauncher`: `infer_nodes` (host math), `_build_appdef` (3-role controller+trainer+generators, whole-host pinning, per-role `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN`), `acquire_hosts` (controller attaches to worker meshes). `_GPUS_PER_HOST_TYPE`, `_REGION_TO_WS_BASE`. |
| `main.py` | Controller entry. `--job-name` = detached MAST controller (attach + spawn proc meshes + run loop); no `--job-name` = local. |
| `run.sh` | On-node entrypoint (every role): platform010 LD_PRELOAD, libcuda/nvshmem symlinks, NCCL/CUDA env, Manifold mount (`MANIFUSE_BUCKET`), warm-storage mount (`FUSE_SRC` → `/mnt/wsfuse`), conda activate, exec. |
| `local.py` | `--local` dev-server entry; mirrors `run.sh` runtime env without MAST-only bits. |
| `config_registry.py` | `mast_rl` recipes → `MastRLConfig` (OSS `Controller.Config` + `launcher`). Only `test_0_6b`, `test_30b_multi_hosts`, `test_rl_grpo_qwen3_30b_a3b_varlen` (all AlphabetSort). |
| `env_defaults.py` | `TRAINING_ENV_DEFAULTS` (alloc conf, NCCL IB/RDMA, Monarch) + `DEEP_EP_ENV` (MNNVL kill-switches, NVSHMEM/IBGDA RoCE GID 3, GIN). |
| `build_conda.sh` | Backup: build local `rlmast` env from source (cu130 torch/vLLM, CUDA 13 toolkit, torchstore, deep_ep, Meta monarch+torchx). Phased out in favor of `--fbpkg`. |
