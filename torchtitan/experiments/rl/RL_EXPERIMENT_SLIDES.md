# TorchTitan RL Experiment
A deep dive into torchtitan/experiments/rl

---
# What is it?
- Experimental RL post-training stack for LLMs, inside TorchTitan
- GRPO / DAPO policy-gradient RL
- Trainer = TorchTitan  |  Generator = vLLM
- ONE unified model definition shared by both
- Monarch = actor orchestration, TorchStore = weight sync
> The unified model enables bitwise trainer/generator agreement.

---
# The core idea
- RL bug trap: generator and trainer are usually different code paths
- Their log-probs drift -> biased importance ratio -> silently wrong training
- Fix: run the SAME TorchTitan model in vLLM and in the trainer
- Then you can verify logprob_diff == 0 (bitwise)

---
# Architecture
- Controller: async orchestrator (train.py / controller.py)
- PolicyTrainer: TorchTitan model, does fwd/bwd/optim
- VLLMGenerator(s): vLLM engine, samples rollouts
- Trainer and generator live on DISJOINT GPU meshes
- Weights flow trainer -> generators via TorchStore

---
# The async loop
- data_input -> rollout[N] -> batcher -> queue -> trainer
- Connected by a shared RolloutGroupWorkBuffer
- Trainer loop is the clock (finite N steps)
- Producers loop forever until the buffer closes

---
# Off-policy control (key knob)
- active_slots = (max_offpolicy_steps + 1) x groups_per_step
- max_offpolicy_steps = 0 -> strict on-policy (needed for bitwise)
- > 0 -> async: generation runs ahead, staleness bounded
- Buffer slot accounting guarantees no sample is "born stale"

---
# Building a task = 4 small files
- data.py: dataset yielding samples
- env.py: a MessageEnv (init + step) in message space
- rubric.py: one or more RewardFn (weighted sum)
- rollouter.py + config_registry.py: wire it together
> Framework handles all token plumbing + failure modes

---
# Losses & advantage
- DAPOLoss: per-token clipped surrogate, asymmetric clip-higher
- GRPOLoss: DAPO with symmetric clip_eps
- Advantage: A = (r - mean) / denom
- Dr.GRPO (mean baseline) or standard GRPO (std-normalized)

---
# Bitwise parity (standout feature)
- DebugConfig(batch_invariant=True, deterministic=True)
- Deterministic mm/attention/NCCL kernels; fp32 RoPE cache
- Both sides compute in bf16 (FSDP mixed precision on trainer)
- Result: logprob_diff/max == 0 every step
- Cost: ~2.5x raw compute, but wall-clock ~ unchanged if orchestration-bound

---
# Two worked examples
- alphabet_sort: smoke test. Sort author names, difflib similarity reward
- search_r1: multi-turn tool-use QA with a search tool, exact-match reward
- search_r1 results: Qwen3-1.7B EM 0.05 -> 0.41; 8B 0.26 -> 0.45

---
# What you can do today
- GRPO/DAPO on Qwen3 0.6B -> 30B MoE, GPT-OSS 20B
- Single- and multi-turn tool-using tasks
- Async off-policy OR strict on-policy
- FSDP + TP + EP, dense + MoE
- Multiple generator replicas with load-balanced routing
- Checkpoint/resume, W&B/TensorBoard metrics, rollout recording

---
# What is missing
- Only GRPO/DAPO (no PPO-critic, no RLHF reward model, no KL penalty)
- No pipeline parallelism in the trainer
- Partial resume (buffer + dataset position not restored)
- Best-of-1 greedy validation only; no periodic eval
- Bitwise parity needs matched TP, no sequence parallelism
- ~110 TODOs across the tree

---
# How you could contribute
- Easy: new task example, new RewardFn, new advantage estimator
- Medium: pluggable filters/packers, best-of-N validation, full resume
- Ambitious: pipeline parallelism, backend-agnostic generator, PPO+critic
- Add tests alongside (the experiment has a real test suite)

---
# Takeaways
- Unified model = the whole point (bitwise agreement)
- Bounded work buffer = the engine (decouple + bound off-policiness)
- Everything is Configurable and pluggable
- Genuinely experimental -> big contribution surface
> Full guide: RL_EXPERIMENT_DEEP_DIVE.md
