# TorchTitan RL Experiment — A Complete Deep Dive

> Location: `torchtitan/experiments/rl/`
> Status: **Active development.** APIs and configs change frequently. This document is a learning guide, not a stable API reference.
> Audience: an engineer who wants to understand *what this is, how it works, what you can do with it, what it's missing, and how to contribute.*

---

## 1. TL;DR — What is this?

This is an **experimental reinforcement-learning (RL) post-training stack for LLMs**, built inside TorchTitan. It does **GRPO-style RL** (Group Relative Policy Optimization, plus its DAPO generalization) on language models, using:

- **TorchTitan** for the *trainer* (the model being optimized, with FSDP/TP/EP parallelism, optimizer, checkpointing).
- **vLLM** for the *generator* (fast rollout/sampling of completions).
- **One unified model definition** shared by both trainer and generator — so the exact same model runs in training and inference. This is the key architectural bet: it enables *bitwise* verification that the generator and trainer agree, killing a whole class of subtle RL correctness bugs.
- **[Monarch](https://github.com/meta-pytorch/monarch)** as the distributed actor framework (orchestrates trainer + generators on separate GPU meshes with async messaging).
- **[TorchStore](https://github.com/meta-pytorch/torchstore)** for weight synchronization from trainer → generators (supports direct GPU-to-GPU RDMA).

The whole thing runs an **async, off-policy-bounded training loop**: generators produce rollouts continuously while the trainer does gradient steps, with a buffer that bounds how "stale" (off-policy) any training sample can be.

### The one-paragraph mental model

> A **Rollouter** turns a dataset prompt into scored **rollouts** (a model plays out a task in an **Environment**, gets rewarded by a **Rubric**). Rollouts become **advantages** (GRPO: reward minus the group mean). A **Controller** runs a pipeline of async loops that feed those rollouts through a **Batcher** into a **PolicyTrainer** (TorchTitan), which does a clipped-surrogate policy-gradient step. After each optimizer step, updated weights are pushed to the **vLLM generators** via TorchStore so the next rollouts are fresh. A bounded **work buffer** keeps generation running ahead of training without letting samples get too old.

---

## 2. Why it exists (the problem it solves)

RL for LLMs has a nasty, well-known correctness trap: **the generator and the trainer are usually two different code paths.** The generator (vLLM, TensorRT, etc.) samples completions and reports log-probs; the trainer (a training framework) recomputes log-probs to compute the policy-gradient ratio `pi_theta / pi_old`. If those two disagree — even by tiny floating-point amounts — the importance ratio is wrong, and you silently train on a biased objective. In MoE models this can even flip expert routing.

This experiment's thesis: **use ONE model definition for both.** TorchTitan's model runs inside vLLM (via a wrapper) *and* as the trainer. That makes it possible to drive the generator/trainer log-prob difference to **exactly zero** (bitwise) under "batch-invariant mode," which is a debugging superpower for RL.

---

## 3. High-level architecture

```diagram
                         +-----------------------------------------+
                         |            Controller (train.py)         |
                         |   async orchestrator of the whole loop   |
                         +---------------+-------------------------+
                                         | owns
            +----------------------------+----------------------------+
            |                            |                            |
            v                            v                            v
   +----------------+          +------------------+        +--------------------+
   |   Rollouter    |          |  PolicyTrainer   |        |  VLLMGenerator(s)  |
   | dataset + env  |          |  (TorchTitan)    |        |   (vLLM engine)    |
   |  + rubric      |          |  fwd/bwd/optim   |        |  sampling/rollouts |
   +-------+--------+          +--------+---------+        +---------+----------+
           | produces scored             | gradient step             | generates
           | RolloutGroups               |                           | completions
           v                             |  weights pushed via        |
   +----------------+                    |  TorchStore ---------------+
   |  Work Buffer   |                    |  (trainer -> generators)
   | bounds off-    |<-------------------+
   | policy staleness|
   +----------------+

   Trainer mesh and Generator mesh(es) live on DISJOINT GPUs.
   Monarch spawns the actors on each mesh and handles async messaging.
```

### The three actors / roles

| Role | Class | Runs on | Job |
|------|-------|---------|-----|
| **Orchestrator** | `Controller` (`controller.py`) | driver process | Runs all async loops; owns trainer + generators + rollouter; coordinates weight sync, validation, metrics. |
| **Trainer** | `PolicyTrainer` (`actors/trainer.py`) | trainer GPU mesh | TorchTitan model. Endpoints: `forward_backward`, `optim_step`, `push_model_state_dict`, `save_checkpoint`. |
| **Generator** | `VLLMGenerator` (`actors/generator.py`) | generator GPU mesh(es) | vLLM engine running the *same* TorchTitan model. Endpoints: `generate`, `pull_model_state_dict`, `start_engine_loop`. |


---

## 4. The async training loop (the heart of it)

The `Controller.run()` method (`controller.py`) launches several `asyncio` loops that form a producer->consumer pipeline connected by a shared **`RolloutGroupWorkBuffer`**:

```diagram
_data_input_loop -> _rollout_loop[N] -> _batcher_loop -> training_batch_queue -> _trainer_loop
        |                  ^                  ^
        v                  |                  |
        +----- RolloutGroupWorkBuffer --------+
```

Loop by loop:

1. **`_data_input_loop`** — pulls one sample from the dataset (`rollouter.get_training_sample()`), wraps it as a `RolloutGroupWork`, and adds it to the buffer. Blocks when there's no free active slot (backpressure). Separated from rollout so slow data prep overlaps generation.

2. **`_rollout_loop[N]`** (N parallel workers, one per active buffer slot) — claims a waiting work item, calls `rollouter.run_group_rollouts(...)` to generate + score a whole GRPO group, records raw rollouts to disk, and finalizes the group back into the buffer. A failed group becomes an empty group + a failure metric (doesn't crash the run).

3. **`_batcher_loop`** — takes the oldest finalized group (strict FIFO), converts it into `TrainingSample`s (`TrainingSampleBuilder`), filters untrainable/zero-variance groups, and accumulates them in the `Batcher`. When enough trainable groups are accumulated, packs a `TrainingBatch` and puts it on the queue.

4. **`_trainer_loop`** — the only *finite* loop. For `num_training_steps` steps: get a packed batch, run `forward_backward` over microbatches, wait for the previous weight push, run `optim_step`, wait for the previous generator pull, then kick off the next async push->pull->slot-release. Also saves checkpoints and logs metrics.

**Shutdown logic:** the trainer loop is the clock. When it finishes N steps, `run()` closes the buffer (waking producers), cancels remaining tasks, then runs post-training validation.

### Off-policy control (the crucial RL knob)

```
max_active_rollout_groups = (max_offpolicy_steps + 1) * num_groups_per_train_step
```

- `max_offpolicy_steps = 0` -> **fully on-policy (sync)**: generator and trainer alternate in lockstep. Required for bitwise parity.
- `max_offpolicy_steps > 0` -> **async**: generation runs ahead by up to that many train-steps. The buffer's active-slot budget guarantees **no sample is ever "born stale."** Policy age is computed at *consumption* time against the live trainer version.

This is arguably the most important design idea in the whole experiment: **the buffer's slot accounting is what bounds off-policiness**, not ad-hoc checks. Slots are released only after the weight pull completes (`release_active_groups(..., "trained")`), so a fresh sample can only enter once the pipeline has advanced.

---

## 5. Core components in detail

### 5.1 Rollouter (`rollout/rollouter.py`)
The RL analog of a `Dataloader`. Given a dataset sample, it:
- builds `group_size` sibling **envs** (`make_env_group`),
- drives each rollout turn-by-turn against the generator (`_run_single_rollout`), alternating env-step <-> generate,
- scores the group with a **Rubric** (`score_group`),
- turns rewards into per-rollout **advantages** (`AdvantageEstimator`).

Subclass it to customize (e.g. cross-sibling judging). Most tasks just supply config: `train_dataset`, `validation_dataset`, `rubric`, `message_env`.

### 5.2 Environment (`environment/message.py`, `environment/token.py`)
Two layers:
- **`MessageEnv`** (abstract, user-written): works purely in *messages*. Implement `init()` (opening messages + tool specs) and `step(completion_message)` (env reply, `done` flag, optional per-step rewards). Never sees tokens. Example in the code: a calculator tool env.
- **`TokenEnv`** (framework-provided wrapper): converts messages <-> tokens via a **Renderer**, enforces limits (`max_rollout_tokens`, `max_num_turns`, `step_timeout_s`), handles parse/length/timeout failures. Keeps the rollout loop clean.

This split is elegant: **you write task logic in message space; the framework handles all the token plumbing and failure modes.**

### 5.3 Rubric & RewardFn (`rubrics/rubric.py`)
- **`RewardFn`**: a single Configurable reward function with a `weight`. Implement `async __call__(rollout, env_input) -> float`.
- **`Rubric`**: holds a list of weighted reward fns, computes a normalized weighted sum. Supports `truncation_reward` / `error_reward` short-circuits. Override `score_group` for cross-sibling scoring (pairwise, diversity, rank-norm).

### 5.4 Advantage (`rollout/advantage.py`)
`A_i = (r_i - mean(r)) / denom`.
- `should_std_normalize=False` (default) -> **Dr.GRPO** (mean-baseline only).
- `should_std_normalize=True` -> **standard GRPO** (divide by group reward std + eps).

### 5.5 Losses (`losses/`)
- **`DAPOLoss`** (`dapo.py`): the base per-token clipped PPO surrogate with **asymmetric "clip-higher"** bounds (`ratio_clip_low`, `ratio_clip_high`). Larger upper bound keeps more mass on up-weighted tokens -> counters entropy collapse ([DAPO paper](https://arxiv.org/abs/2503.14476)). Drops tokens with non-finite generator log-probs (rather than training them as if on-policy). Clamps `|log ratio|` to 10.0 to prevent `exp()` overflow.
- **`GRPOLoss`** (`grpo.py`): just DAPO with symmetric clip (`clip_eps` for both bounds).
- Loss is `sum(per-token loss) / num_global_valid_tokens`, so gradient accumulation across microbatches matches one big batch.
- Rich metrics: ratio-clipped fraction, log-prob diff (the k1 MC estimate of -KL), entropy, generator-logprob-NaN fraction.

### 5.6 PolicyTrainer (`actors/trainer.py`)
A Monarch `Actor` wrapping TorchTitan's training machinery: model build, optimizer, LR scheduler, FSDP/TP/EP parallelism, checkpointing, and the loss. Split endpoints (`forward_backward` then `optim_step`) let the controller interleave weight sync. Casts weights to the generator's dtype on `push_model_state_dict`. **No pipeline parallelism yet** (asserts exactly one model part).

### 5.7 VLLMGenerator (`actors/generator.py`) — the most complex file (~1500 lines)
A Monarch `Actor` wrapping a vLLM V1 engine running the TorchTitan model. Key ideas:
- **Continuous batching via a decoupled engine loop.** Rank 0 accepts `generate` calls, enqueues a `LoopDecision`, and awaits a future. A background `_engine_loop` per rank consumes decisions and runs `engine.step()` bursts (`max_engine_steps_between_decisions`, default 16). New requests join mid-flight instead of waiting for the batch to drain.
- **SPMD across TP/DP ranks:** rank 0 drives followers via `broadcast_object_list`. A `RequestDispatcher` routes requests to DP ranks and collects results per-request (not per-batch).
- **Weight sync rides the same loop:** `pull_model_state_dict` queues a `PULL_MODEL_STATE_DICT` decision applied between step bursts. Supports "hot-swap" (no drain) or drain-first.
- **Prefix-cache correctness:** `reset_prefix_cache_on_weight_sync` / `reset_running_requests_on_weight_sync` prevent reusing KV computed under old weights.

### 5.8 Weight sync (`components/weight_sync.py`)
`WeightSyncManager` **overlaps the trainer->generator handoff with the next training step.** Push after `optim_step`, pull after push, release buffer slots after pull — all in the background, awaited just-in-time so weights aren't mutated mid-use.

### 5.9 Work buffer (`components/work_buffer.py`)
`RolloutGroupWorkBuffer`: the run-ahead buffer. Entries move `WAITING -> INFLIGHT -> FINALIZED`. Crucially, a slot is **not** released when finalized/taken — only on explicit `release_active_groups` calls (by trainer after pull, or batcher on untrainable groups). This is what guarantees the pipeline never exceeds `max_active_rollout_groups` and never produces born-stale examples.

### 5.10 Batcher & TrainingSampleBuilder (`components/`)
- **`TrainingSampleBuilder`**: RolloutGroup -> data-validity filters (drop empty/zero-std-reward groups) -> per-turn `TrainingSample`s with token_ids, loss_mask, generator logprobs, advantages.
- **`Batcher`**: next-fit packs training samples into microbatches respecting `seq_len` and `local_batch_size x dp_degree`; computes `num_global_valid_tokens` for loss normalization.

### 5.11 Routing (`routing/`)
Two layers of load balancing, sharing the same `RoutingStrategy` classes:
- **`InterGeneratorRouter`** (controller side): routes a call across generator *meshes* (replicas).
- **`IntraGeneratorRouter`** (rank-0 side): routes a request across *DP ranks* within one mesh.
Strategies include `LeastLoadedRoutingStrategy` and `StickySessionRoutingStrategy` (a sample's turns reuse one generator's prefix KV cache — big perf win for multi-turn).

### 5.12 Observability (`observability/metrics/`, `controller_metrics.py`, `rollout_recorder.py`)
- Typed `Metric(key, value)` records with lazy reduction (Mean/Sum/Max/NoReduce), logged to console + **W&B** (on by default) + **TensorBoard**.
- `RolloutSampleRecorder` dumps raw rollouts to JSONL for inspection/debugging (kept even for dropped groups).
- Structured trace-span logging (`sl.log_trace_span`) throughout.


---

## 6. The unified model & bitwise parity (the standout feature)

### Unified model definition
The same TorchTitan model spec builds both the trainer's model and (through `models/vllm_wrapper.py` -> `TorchTitanVLLMModel`) the model running inside vLLM. Benefits: fast iteration, shared optimizations, and — the big one — **straightforward bitwise parity verification.**

### Batch-invariant mode (`batch_invariance.py`, `docs/bitwise_parity.md`)
The problem: the generator computes logprobs in one batch composition (e.g. 8 completions); the trainer recomputes them in another (e.g. 2 after DP sharding). Floating-point accumulation order differs -> logprobs drift -> biased importance ratio.

The fix, enabled by `DebugConfig(batch_invariant=True, deterministic=True)`:
- **Op overrides** (`set_batch_invariance` in `torchtitan/distributed/utils.py`): swap `mm`/`addmm`/`log_softmax`/`mean.dim` for [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) Triton kernels with fixed tile order; force flash-attention `num_splits=1`; force deterministic single-channel tree NCCL (matching vLLM); disable TF32 and reduced-precision reductions.
- **Generator-only patches** (`batch_invariance.py`): route vLLM's fused attention back through TorchTitan's own kernel (Varlen/Flex/GPT-OSS), patch `bmm` for the MoE router gate, and replace vLLM's fused logprob kernel with the trainer's `compute_logprobs`.
- **RoPE cache dtype**: keep `freqs_cis` in fp32 across FSDP boundaries.
- **Precision**: both sides compute in bf16. The trainer keeps fp32 master weights and FSDP mixed precision casts to bf16 for the forward — bitwise identical to the generator's bf16 forward, even at `data_parallel_shard_degree=1`.

Result: `bit_wise/logprob_diff/max == 0` at every step. Measured cost (Qwen3-8B, Search-R1, 8xH100): raw compute ~2.4-2.9x slower, but **end-to-end wall-clock ~ unchanged** for orchestration-bound workloads.

**Current parity limitations:** trainer and generator must use the **same TP degree**; **sequence parallelism is not supported** in batch-invariant mode (reduce-scatter lacks a deterministic tree impl).

---

## 7. The two worked examples

### 7.1 `alphabet_sort` — the smoke-test / hello-world
- **Task**: model is shown shuffled arXiv author names, must return them sorted alphabetically (by first or last name) inside an `<alphabetical_sorted>` block. Multi-turn variants add names each turn and ask for a re-sort marking new names.
- **Reward** (`RewardAlphabetSort`): difflib order-similarity raised to a power (partial credit -> smooth learning signal).
- **Purpose**: the end-to-end environment/setup smoke test. Run it first to verify your install.
- Ships **many configs** (`config_registry.py`): Qwen3-0.6B/1.7B/14B (varlen & flex), GPT-OSS-20B, MoE debug (EP+TP), DeepEP, and batch-invariant variants.

Run:
```bash
python -m torchtitan.experiments.rl.train --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen
```

### 7.2 `search_r1` — the flagship multi-turn tool-use recipe
- **Task**: open-domain QA. The model gets a `search` tool; it calls search when it needs facts, gets retrieved wiki passages back as a `tool` message, and answers. Reward is **exact-match (EM)** against golden answers.
- **Anti-reward-hacking levers**: `no_search_penalty` (a correct answer that never searched scores less) and `retrieval_score` (a wrong answer gets partial credit if search surfaced the golden answer). These "put search on the gradient."
- **Infra**: needs a local dense-retrieval server (e5 index over wiki-18) on spare GPUs; data streamed from HF `PeterJinGo/nq_hotpotqa_train`.
- **Results** (from the README): Qwen3-1.7B EM ~0.05 -> ~0.41; Qwen3-8B EM ~0.26 -> ~0.45.
- Runs entirely on the framework's generic multi-turn rollouter + continuous-batching generator — the only task-specific code is the folder + config.

Run:
```bash
python torchtitan/experiments/rl/train.py --module search_r1 --config rl_grpo_qwen3_1_7b_search_r1
```

---

## 8. What you can DO with it (capabilities today)

- **GRPO and DAPO** RL post-training on real LLMs (Qwen3 0.6B -> 30B-A3B MoE, GPT-OSS 20B).
- **Single-turn and multi-turn, tool-using** tasks (search, calculators, any `MessageEnv`).
- **Async off-policy** training with a bounded staleness window, *or* strict on-policy.
- **Dense and MoE** models, with FSDP + TP + EP; multiple attention backends (varlen FA3, flex, DeepEP for MoE).
- **Bitwise-reproducible** trainer/generator numerics for debugging.
- **Multiple generator replicas** with inter/intra load-balancing + sticky-session prefix-cache reuse.
- **Overlapped weight sync** (TorchStore, GPU-to-GPU RDMA capable) hidden behind the training step.
- **Checkpoint/resume** (model + optimizer + policy_version) via TorchTitan's CheckpointManager.
- **Rich metrics** (W&B / TensorBoard), rollout recording (JSONL), held-out validation with pre/post reward delta.
- **Pluggable everything** via the `Configurable` pattern: datasets, envs, rubrics, reward fns, advantage estimators, routing strategies, losses.

### To build your own task, you write ~4 small files (see `examples/`):
1. `data.py` — a dataset that yields samples.
2. `env.py` — a `MessageEnv` subclass (`init` + `step`).
3. `rubric.py` — one or more `RewardFn`s.
4. `rollouter.py` + `config_registry.py` — wire them into a `Rollouter.Config` / `Controller.Config`.
Then run `--module <your_module> --config <your_config>`.


---

## 9. What it's MISSING (limitations & gaps)

These come straight from the code's own `TODO`/limitation comments and docstrings (~110 TODO markers in the RL tree). Grouped by theme:

### Algorithms / RL features
- **Only GRPO/DAPO losses.** No PPO with a learned value/critic, no vanilla policy gradient, no reward-model-in-the-loop RLHF, no KL-to-reference-policy penalty term (only a KL *metric*). No offline/DPO-style methods.
- **Advantage estimation is group-relative only** (GRPO/Dr.GRPO). No GAE, no per-token value baselines.
- **Advantage/filters not yet pluggable** — group-level filters (drop zero-std, etc.) are hard-coded; a TODO wants an ordered user-pluggable filter list.

### Parallelism / scale
- **No pipeline parallelism** in the trainer (asserts a single model part).
- **Bitwise parity requires matched TP** and **forbids sequence parallelism**.
- Single rollouter only — **no data mixing** across multiple datasets/rollouters yet (TODO).

### Resume / durability
- **Resume is partial**: only model/optimizer/policy_version are restored. The **in-flight rollout buffer and dataset stream position are NOT** — a resumed run refills the buffer and re-reads data from the start. (Two explicit `TODO(resume)`s.)
- No warm-start (cold start fills every active slot at step 0).

### Generator / inference
- **Per-token policy-version attribution is approximate** — only per-turn min/max policy version is tracked, not exact per-token (would need `RequestOutputKind.CUMULATIVE`).
- vLLM `EngineConfig` isn't fully exposed as a config field yet.
- `generator.py` is ~1500 lines — a TODO wants a backend-agnostic `BaseGenerator` split so non-vLLM backends can plug in.
- MoE all-to-all path blocks CUDA-graph/compile in some configs (worked around by disabling them).

### Validation / eval
- Validation is **best-of-1 greedy only** (no best-of-N, no pass@k) and runs only pre/post training — **no periodic validation** during training yet.
- Fixed `num_samples` — can't yet say "run the whole eval set."

### Batching / streaming
- **Can't stream microbatches** (interleave pack->train) because the loss needs `num_global_valid_tokens` up front.
- Packing is greedy next-fit and not pluggable (TODO wants a `Packer` protocol: best-fit, etc.).
- Buffer slots released in batches by a single-producer data loop — a noted perf bottleneck.

### Misc
- Renderer owns its own tokenizer (can't yet bring-your-own-tokenizer; blocked on upstream `renderers`).
- Naming debt (`sample` overloaded), module-path debt (`controller_metrics` placement), and several perf TODOs (broadcast double-serialization, staggered per-generator fetches, prefix-cache salting per new rollout).

---

## 10. What YOU could add / contribute

Ordered roughly easy -> ambitious. These map directly onto the gaps above and the code's own TODOs.

### Good first contributions
1. **A new task example.** Add a folder under `examples/` (math reasoning like GSM8K, code-gen with unit-test reward, a game). This is the highest-leverage way to learn the stack — you touch dataset, env, rubric, config, and tests, but nothing in the core loop.
2. **A new `RewardFn`.** LLM-as-judge reward, format/length rewards, a reward model wrapper. Self-contained.
3. **A new advantage estimator.** Add leave-one-out (RLOO) or a different baseline alongside GRPO in `advantage.py`.
4. **Periodic validation.** The `ValidationConfig` explicitly has a `# TODO: enable periodic validation` — wire it into the trainer loop with proper overlapping.

### Medium
5. **Pluggable group-level filters.** Generalize `TrainingSampleBuilder`'s hard-coded filters into an ordered, user-configurable filter list (the code asks for this).
6. **Pluggable packing (`Packer` protocol).** Add best-fit / other packers to the `Batcher`.
7. **best-of-N / pass@k validation.** Extend `_collect_validation_rollouts` beyond greedy best-of-1.
8. **Full resume.** Persist dataset stream position + in-flight buffer so a restarted run truly continues (two open TODOs).
9. **A KL-to-reference penalty** term in the loss (a real KL regularizer, not just the metric), with a frozen reference policy.

### Ambitious
10. **Pipeline parallelism** support in `PolicyTrainer` (currently blocked by the single-model-part assertion).
11. **A backend-agnostic `BaseGenerator`** so non-vLLM inference engines can be plugged in (the generator TODO).
12. **A second RL algorithm** (PPO-with-critic, or a value-based method) implemented against the existing loss/actor interfaces.
13. **Multi-rollouter data mixing** (the `# TODO: support multiple rollouters`).
14. **Extend bitwise parity** to sequence parallelism / mismatched TP (hard — needs deterministic reduce-scatter).

### Where to look before contributing
- `CONTRIBUTING.md` at repo root (this is `pytorch/torchtitan`).
- The `.claude/skills/inference_perf_hillclimb/SKILL.md` inside the RL dir — a profiler-driven methodology for closing the inference gap to native vLLM. Great if you care about generator throughput.
- Tests in `tests/` — the experiment has real unit tests (batcher backpressure, router, metrics, bitwise parity, both examples). Add tests with your contribution.


---

## 11. How to run it (quick start recap)

**Prereqs** (from the RL `README.md`): a CUDA GPU box (H100/H200 ideal for FA3; A100 falls back to FA2), Python 3.12 via `uv`.

```bash
# 1. Environment
pip install uv
uv venv --python 3.12 titan-rl && source titan-rl/bin/activate

# 2. Monarch, TorchStore, Renderers
uv pip install torchmonarch
uv pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install pygtrie portpicker
uv pip install "git+https://github.com/PrimeIntellect-ai/renderers.git@main"

# 3. Flash Attention 3 (H100+)
uv pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130

# 4. (optional) batch-invariant ops for bitwise mode
uv pip install --no-deps "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"

# 5. PyTorch nightly + vLLM + torchcomms
uv pip install torch vllm torchcomms --pre \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --index-strategy unsafe-best-match

# 6. PYTHONPATH so Monarch workers import local torchtitan
cd <your_torchtitan_root> && export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 7. Download a checkpoint
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B \
  --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...

# 8. Smoke test (also verifies your setup end-to-end)
python -m torchtitan.experiments.rl.train --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen
```

- **Metrics**: W&B on by default (`wandb login`, or `--metrics.no-enable-wandb`); TensorBoard via `--metrics.enable-tensorboard`.
- **Different checkpoint path**: `--hf_assets_path=<path>`.
- **Bitwise/on-policy debugging**: use a `*_batch_invariant` config.

### GPU budgets (from config docstrings)
| Config | GPUs | Layout |
|--------|------|--------|
| `rl_grpo_qwen3_0_6b_varlen` | 6 | 4 gen (TP4) + 2 train (TP2) |
| `rl_grpo_qwen3_0_6b_flex` | 4 | 2 gen + 2 train |
| `rl_grpo_qwen3_1_7b` | 6 | 4 gen + 2 train |
| `rl_grpo_qwen3_14b` | 16 | 8 gen (TP8) + 8 train (TP8) |
| `rl_grpo_qwen3_30b_a3b_varlen` | 8 | 4 gen + 4 train (TP2 + EP4) |
| `rl_grpo_qwen3_1_7b_search_r1` | 8 | 4 gen (TP4) + 1 train + retrieval server |

---

## 12. File map (cheat sheet)

```diagram
experiments/rl/
|-- train.py                 # entrypoint: provisions meshes, builds Controller, runs
|-- controller.py            # async orchestrator + all the loops (the heart)
|-- controller_metrics.py    # metric helpers for the loop
|-- types.py                 # Completion, TrainingSample, TrainingBatch, RolloutTurnID, ...
|-- batch_invariance.py      # generator-side batch-invariant patches
|-- rollout_recorder.py      # JSONL dump of raw rollouts
|-- renderer.py              # message<->token renderer config
|-- generate.py              # standalone generation / inference benchmark harness
|-- actors/
|   |-- trainer.py           # PolicyTrainer (TorchTitan) Monarch actor
|   +-- generator.py         # VLLMGenerator (vLLM) Monarch actor  (~1500 LoC)
|-- losses/{grpo,dapo}.py    # clipped-surrogate policy-gradient losses
|-- rollout/
|   |-- rollouter.py         # Rollouter: dataset+env+rubric -> scored rollouts
|   |-- advantage.py         # GRPO / Dr.GRPO advantage
|   +-- types.py             # Rollout, RolloutGroup, RolloutTurn, RolloutStatus
|-- rubrics/rubric.py        # Rubric + RewardFn
|-- environment/{message,token}.py  # MessageEnv (user) + TokenEnv (framework wrapper)
|-- components/
|   |-- work_buffer.py       # RolloutGroupWorkBuffer (off-policy bound)
|   |-- weight_sync.py       # WeightSyncManager (overlapped push/pull)
|   |-- batcher.py           # packs TrainingSamples -> microbatches
|   +-- training_sample_builder.py  # RolloutGroup -> TrainingSamples + filters
|-- routing/                 # inter/intra generator routing strategies
|-- models/                  # vllm_wrapper, vllm_registry, attention, cast_linear
|-- observability/metrics/   # typed Metric records -> console/W&B/TensorBoard
|-- examples/
|   |-- alphabet_sort/       # smoke-test task (many configs)
|   +-- search_r1/           # flagship multi-turn tool-use QA recipe
|-- docs/bitwise_parity.md   # deep dive on numerics/determinism
|-- tests/                   # unit + integration tests
+-- README.md                # setup + quick start
```

---

## 13. Key takeaways

1. **The unified model is the whole point** — one TorchTitan model runs as both trainer and vLLM generator, enabling bitwise trainer/generator agreement and killing a class of silent RL bugs.
2. **The async loop with a bounded work buffer** is the engine — it decouples generation from training while provably bounding off-policiness.
3. **Everything is `Configurable` and pluggable** — datasets, envs, rubrics, rewards, advantages, routing, losses. Building a new task is ~4 small files.
4. **It's genuinely experimental** — ~110 TODOs, partial resume, GRPO/DAPO only, no PP, best-of-1 validation. That's the contribution surface, not a criticism.
5. **Two examples anchor it**: `alphabet_sort` (smoke test) and `search_r1` (real multi-turn tool-use with published EM curves).

---

*Generated as a learning guide for the TorchTitan RL experiment. Cross-check against the live code — this area moves fast.*
