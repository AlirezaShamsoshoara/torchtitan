# Report: Attention Residuals for Qwen3 (MoE)

## Status: PLANNING

Implementation has not started. This document will be updated as tasks are
completed with results, findings, and analysis.

---

## Research Findings

### Paper Analysis

The Attention Residuals paper (Kimi Team, 2025) conducted **all experiments on
MoE models** (Kimi Linear, 48B/3B activated MoE). Key findings relevant to
this Qwen3 experiment:

- **1.25x compute advantage** at scale (5.6 PFLOP/s-days)
- **<4% TPS overhead** under pipeline parallelism with cross-stage caching
- Block AttnRes with N=8 blocks recovers most of Full AttnRes gains
- Architecture is agnostic to dense vs MoE --- treats all layer outputs equally

### Existing Llama3 AttnRes Results (for reference)

| Scale | AttnRes vs Llama3 | TPS Overhead | Memory Overhead |
|-------|-------------------|-------------|-----------------|
| debugmodel (6L) | Llama3 wins (-8.9%) | 17% | 1.6% |
| debugmodel_v2 (32L) | AttnRes wins (96.6% steps) | 28% | ~0% |
| 1B (full C4) | AttnRes wins (+1.0%) | 36% | 0.2% |
| 8B (full C4) | Llama3 wins (-3.1%) | 30% | 0.03% |

Key takeaway: AttnRes benefits increase with depth. The debugmodel (6 layers)
is too shallow; 32+ layers shows clear advantage.

### Qwen3 Architecture Assessment

**Dense variants** (0.6B-32B):
- Architecturally similar to Llama3 (pre-norm, SwiGLU FFN, GQA)
- Key differences: QK normalization, cos_sin RoPE, head_dim=128
- None of these affect AttnRes (it operates on block-level hidden states)
- Weight tying used for models up to 4B (AttnRes supports this)

**MoE variants** (30B-A3B, 235B-A22B):
- Replace dense FFN with TokenChoiceTopKRouter + GroupedExperts
- 128 experts, top-8 selection, softmax scoring, no shared experts
- MoE output has same shape as dense FFN output (B, T, D)
- AttnRes sees MoE as a black box --- no interference expected

**Parallelism complexity**:
- Qwen3 MoE uses EP (Expert Parallelism) in addition to FSDP/TP
- EP is internal to the MoE layer (token distribution among experts)
- AttnRes operates on the hidden state before/after MoE, not within it
- No EP-specific AttnRes modifications expected

### Code Reuse Assessment

| Component | Reuse from | Modification needed |
|-----------|-----------|-------------------|
| `block_attn_res` function | `attn_residuals/attn_res.py` | None (import directly) |
| `_ensure_dtensors` | `attn_residuals/attn_res.py` | None (imported with block_attn_res) |
| MoE layer | `torchtitan/models/common/moe/` | None (used by Qwen3 unchanged) |
| Non-MoE TP plan | `qwen3/parallelize.py` | Extend with AttnRes entries |
| MoE EP/TP | `llama4/parallelize.py` | None (reuse as-is) |
| FSDP wrapping | `llama4/parallelize.py` | None (reuse as-is) |
| Pipeline parallelism | `torchtitan/distributed/pipeline_parallel.py` | Need PP-compatible forward via block caching |
| State dict adapter | `qwen3/state_dict_adapter.py` | Not needed (no pretrained AttnRes weights) |

### Pipeline Parallelism Analysis

**Problem**: AttnRes threads a growing `blocks` list through the forward pass.
PP requires fixed-size tensor communication between stages via `dist.isend/irecv`.
Three incompatibilities:
1. Variable-length list cannot be a fixed-size send/recv buffer
2. Lists are not tensors (`PipelineStage` rejects non-tensor outputs)
3. `partial_block` can be `None` at block boundaries

**Solution**: Block caching (paper Section 4.1). Each stage:
- Receives a single `prev_block` tensor from prior stage (same as standard PP)
- Maintains its own local `blocks` list, seeded with `prev_block`
- Sends a single tensor (last completed block) to next stage

**Approaches evaluated**:

| Approach | Complexity | Overhead | Numerical equiv? |
|----------|-----------|---------|-------------------|
| A. Pack/unpack (pad to max blocks) | Medium | Bandwidth waste | Exact |
| B. Fixed-size tuple with zeros | Medium | Same waste | Exact |
| **C. Block caching (paper)** | High | **<4% (paper)** | **Exact if full; approx if incremental** |
| D. Custom PipelineStage | Very High | Minimal | Exact |

**Selected**: Option C (block caching). Matches paper's design, enables the
claimed <4% overhead, and is architecturally clean.

---

## Results

*To be populated as experiments run.*

### Dense debugmodel

| Metric | AttnRes Qwen3 | Qwen3 Baseline | Delta |
|--------|--------------|----------------|-------|
| Final loss (500 steps) | -- | -- | -- |
| TPS (tokens/sec) | -- | -- | -- |
| Peak memory (GB) | -- | -- | -- |

### MoE debugmodel_moe

| Metric | AttnRes Qwen3 | Qwen3 Baseline | Delta |
|--------|--------------|----------------|-------|
| Final loss (500 steps) | -- | -- | -- |
| TPS (tokens/sec) | -- | -- | -- |
| Peak memory (GB) | -- | -- | -- |

### 0.6B Dense

| Metric | AttnRes Qwen3 | Qwen3 Baseline | Delta |
|--------|--------------|----------------|-------|
| Final loss (1000 steps) | -- | -- | -- |
| TPS (tokens/sec) | -- | -- | -- |
| Peak memory (GB) | -- | -- | -- |

### 30B-A3B MoE

| Metric | AttnRes Qwen3 | Qwen3 Baseline | Delta |
|--------|--------------|----------------|-------|
| Final loss (5000 steps) | -- | -- | -- |
| TPS (tokens/sec) | -- | -- | -- |
| Peak memory (GB) | -- | -- | -- |

---

## Analysis

*To be populated after experiments.*

### Key Questions to Answer

1. Does AttnRes help MoE models more, less, or equally compared to dense?
2. Is TPS overhead larger with MoE (compounding) or similar (independent)?
3. Does the 1.25x compute advantage from the paper hold in torchtitan?
4. How does EP interact with AttnRes determinism?
5. At what depth/scale does AttnRes become beneficial for Qwen3?
6. Does PP block caching reduce TPS overhead toward <4% (paper's claim)?
7. Does PP + EP + AttnRes compose without correctness issues?

### Comparison with Llama3 AttnRes

*Side-by-side comparison of AttnRes effectiveness on dense (Llama3) vs
MoE (Qwen3) will be added here.*
