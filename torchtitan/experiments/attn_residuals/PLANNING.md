# Attention Residuals (AttnRes) Integration into TorchTitan

## 1. Overview

### What is Attention Residuals?

Standard residual connections in Transformers accumulate all layer outputs with fixed unit weights ($h_l = h_{l-1} + f_{l-1}(h_{l-1})$). This causes:
- **PreNorm dilution**: hidden-state magnitudes grow as O(L), diluting each layer's contribution
- **No selective access**: each layer only sees the aggregated state, not individual prior outputs
- **Output growth**: deeper layers must produce increasingly large outputs to remain influential

**Attention Residuals (AttnRes)** replaces fixed accumulation with learned, input-dependent softmax attention over depth:

```
h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
```

where `alpha_{i->l}` are softmax attention weights computed from a single learned pseudo-query `w_l ∈ R^d` per layer. Keys are `RMSNorm(v_i)` where `v_i` are previous layer outputs (or token embedding for i=0).

### Block AttnRes (Practical Variant)

Full AttnRes requires O(Ld) memory. **Block AttnRes** groups L layers into N blocks (~8):
- **Intra-block**: standard residual accumulation within each block
- **Inter-block**: softmax attention over N block-level representations

This reduces memory from O(Ld) to O(Nd) while recovering most of the gains.

### Key Results from Paper
- Block AttnRes matches loss of baseline trained with **1.25x more compute**
- Training overhead is marginal (<4% with PP, negligible without)
- Inference overhead <2%
- Improvements across all benchmarks (MMLU +1.1, GPQA-Diamond +7.5, HumanEval +3.1)

---

## 2. Current TorchTitan Architecture (Relevant Parts)

### Residual Connection Implementation

The residual connections are in each model's `TransformerBlock.forward()`:

**Llama3** (`torchtitan/models/llama3/model.py:48-59`):
```python
def forward(self, x, freqs_cis, attention_masks, positions=None):
    h = x + self.attention(self.attention_norm(x), freqs_cis, attention_masks, positions)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
```

**Llama4** (`torchtitan/models/llama4/model.py:117-131`): identical pattern, with MoE variant.

**Base Decoder** (`torchtitan/models/common/decoder.py:141-155`):
```python
def forward(self, tokens, attention_masks=None, positions=None):
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
    for layer in self.layers.values():
        h = layer(h, self.freqs_cis, attention_masks, positions)
    h = self.norm(h) if self.norm is not None else h
    output = self.output(h) if self.output is not None else h
    return output
```

### Key Observations
1. Each `TransformerBlock` receives only `h` (the accumulated hidden state) — no access to prior layer outputs
2. The `Decoder.forward()` iterates through layers sequentially, passing a single tensor
3. All model variants (llama3, llama4, qwen3, deepseek_v3, gpt_oss) follow this same pattern
4. Layers are stored in `ModuleDict` keyed by string indices

### Parallelism Infrastructure
- **FSDP** (`torchtitan/distributed/fsdp.py`): wraps each `TransformerBlock` independently with `fully_shard()`
- **Tensor Parallel** (`torchtitan/models/llama3/parallelize.py`): applies TP plans per-block (Colwise/Rowwise parallel for attention and FFN weights, SequenceParallel for norms)
- **Pipeline Parallel** (`torchtitan/distributed/pipeline_parallel.py`): splits model by module FQN, each stage gets a subset of layers + optional tok_embeddings/norm/output
- **Activation Checkpointing** (`torchtitan/distributed/activation_checkpoint.py`): applied per-TransformerBlock via `checkpoint_wrapper`
- **Context Parallel**: applied to inner attention modules

---

## 3. Implementation Plan

### Phase 0: Placement Decision

Per TorchTitan conventions:
- This is an **experiment** (new research idea, not proven in TorchTitan) → belongs in `torchtitan/experiments/attn_residuals/`
- Should NOT modify core torchtitan code (no changes to `torchtitan/models/common/decoder.py` or `torchtitan/models/llama3/model.py`)
- Should reuse TorchTitan's config system

### Phase 1: Core AttnRes Module

**File: `torchtitan/experiments/attn_residuals/attn_res.py`**

```python
class BlockAttnRes(nn.Module):
    """Block Attention Residuals: inter-block attention over block representations."""

    def __init__(self, dim: int):
        super().__init__()
        # Learned pseudo-query w_l ∈ R^d (one per usage site, not per layer)
        self.proj = nn.Linear(dim, 1, bias=False)  # projects keys to logits
        self.norm = RMSNorm(dim)

    def forward(self, blocks: list[Tensor], partial_block: Tensor) -> Tensor:
        """
        Args:
            blocks: list of N completed block representations [B, T, D]
            partial_block: current intra-block partial sum [B, T, D]
        Returns:
            h: attended representation [B, T, D]
        """
        V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
        K = self.norm(V)
        logits = torch.einsum('d, n b t d -> n b t', self.proj.weight.squeeze(), K)
        h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
        return h
```

Key design decisions:
- **Initialize pseudo-query vectors to zero** (critical per paper §5) → uniform initial weights across sources, prevents training volatility
- Each TransformerBlock gets **two** AttnRes instances: one before attention, one before MLP (matching the paper's approach where each sub-layer is a separate "layer")
- Use RMSNorm on keys to prevent magnitude-dominant layers from dominating attention

### Phase 2: Modified TransformerBlock

**File: `torchtitan/experiments/attn_residuals/model.py`**

Create `AttnResTransformerBlock` that wraps an existing block's components but changes the forward signature:

```python
class AttnResTransformerBlock(TransformerBlock):
    """TransformerBlock with Block Attention Residuals."""

    def __init__(self, config, *, layer_id, dim, n_layers, block_size):
        super().__init__()
        # Standard components (attention, feed_forward, norms)
        self.attention = config.attention.build(dim=dim)
        self.feed_forward = config.feed_forward.build(dim=dim)
        self.attention_norm = config.attention_norm.build(normalized_shape=dim)
        self.ffn_norm = config.ffn_norm.build(normalized_shape=dim)

        # AttnRes components (two per layer: pre-attn and pre-MLP)
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_proj = nn.Linear(dim, 1, bias=False)
        self.mlp_res_norm = RMSNorm(dim)

        self.layer_id = layer_id
        self.block_size = block_size  # num sub-layers per block

    def forward(self, blocks, partial_block, freqs_cis, attention_masks, positions=None):
        # Pre-attention AttnRes
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # Block boundary check (block_size counts attn + MLP; each layer has 2)
        if self.layer_id % (self.block_size // 2) == 0:
            blocks = blocks + [partial_block]  # immutable append for grad safety
            partial_block = None

        # Self-attention
        attn_out = self.attention(self.attention_norm(h), freqs_cis, attention_masks, positions)
        partial_block = partial_block + attn_out if partial_block is not None else attn_out

        # Pre-MLP AttnRes
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        # MLP
        mlp_out = self.feed_forward(self.ffn_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block
```

### Phase 3: Modified Decoder

**File: `torchtitan/experiments/attn_residuals/model.py`**

```python
class AttnResDecoder(Decoder):
    """Decoder with Block Attention Residuals."""

    def forward(self, tokens, attention_masks=None, positions=None):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        # Initialize: token embedding is the first "block"
        blocks = [h]
        partial_block = h  # Will be reset at first block boundary

        for layer in self.layers.values():
            blocks, partial_block = layer(blocks, partial_block, self.freqs_cis, attention_masks, positions)

        # Final output uses the last partial_block (or could do final attn_res)
        h = partial_block
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
```

### Phase 4: Config & Registration

**File: `torchtitan/experiments/attn_residuals/config_registry.py`**

Register AttnRes model configs (mirroring Llama3 configs but with added `block_size` parameter):

```python
@dataclass
class AttnResConfig:
    num_blocks: int = 8  # N in the paper
    # block_size is derived: block_size = 2 * n_layers // num_blocks
```

### Phase 5: Parallelism Support

**File: `torchtitan/experiments/attn_residuals/parallelize.py`**

---

## 4. Risk Assessment: Parallelism Interactions

### 4.1 FSDP (Data Parallelism) — LOW RISK

**Current**: Each `TransformerBlock` is wrapped with `fully_shard()` independently.

**Impact**: AttnRes adds 2 small parameters per layer (`attn_res_proj.weight` [d,1] and `mlp_res_proj.weight` [d,1]) plus 2 RMSNorm modules. These are negligible additions and will be sharded along with the rest of the block.

**Key concern**: The `blocks` list passed between layers contains references to tensors from *prior* FSDP-wrapped modules. Since these are activations (not parameters), FSDP should handle them correctly — the parameters are already all-gathered when the block produces its output. No additional all-gather is needed to access block representations in later layers.

**Verdict**: Should work with minimal changes. The `apply_fsdp` function wraps per-TransformerBlock, which will naturally include the new AttnRes parameters.

### 4.2 Tensor Parallelism — MEDIUM RISK

**Current**: TP shards attention weights (Colwise/Rowwise) and FFN weights, with SequenceParallel for norms. Hidden states are sharded on sequence dimension between blocks.

**Impact**: The `block_attn_res` operation involves:
1. `torch.stack(blocks + [partial_block])` — stacking tensors already in Shard(1) (sequence-dim) layout
2. `self.norm(V)` — RMSNorm, which is element-wise (SequenceParallel-compatible)
3. `torch.einsum('d, n b t d -> n b t', proj.weight, K)` — the proj weight is [d] and needs to be Replicated on TP mesh
4. `softmax(0)` — over depth dimension, independent per-token, TP-safe
5. `torch.einsum('n b t, n b t d -> b t d', ...)` — weighted sum, TP-safe if V is properly placed

**Key concern**: The pseudo-query projection `proj.weight` is a [d]-dimensional vector. Under TP, the hidden states between blocks are Shard(1) on sequence dim. The projection `w^T @ RMSNorm(v)` sums over the d-dimension. If d is not sharded (which it isn't — TP shards n_heads, not d), this is safe.

However, the hidden state between `attention.wo` (RowwiseParallel, output Shard(1) on seq dim) and the next norm (SequenceParallel) is in Shard(1) layout. The AttnRes operation happens at this boundary. We need to ensure the AttnRes computation is consistent with SequenceParallel placement.

**Mitigation**:
- Add `SequenceParallel()` plan for `attn_res_norm` and `mlp_res_norm`
- The `proj` weight should be wrapped with `NoParallel` or left as-is (it's a tiny [d,1] linear — replicated computation is fine)
- Alternatively, apply AttnRes in the replicated domain (after the SP→Replicate transition at `PrepareModuleInput`)

**Verdict**: Requires careful TP plan design but fundamentally compatible.

### 4.3 Pipeline Parallelism — HIGH RISK (Primary Challenge)

**Current**: PP splits the model into stages by module FQN. Each stage gets a contiguous set of layers. The `PipelineStage` transfers a single hidden state tensor between stages.

**Impact**: This is the **hardest** interaction. With AttnRes, each layer needs access to block representations from *all previous blocks*, including those on earlier PP stages. This means:

1. **Cross-stage state**: Block representations must be transmitted between PP stages
2. **Growing communication**: As the pipeline progresses, more block representations accumulate
3. **Modified forward signature**: Layers now return `(blocks, partial_block)` instead of just `h`

**The paper addresses this directly (§4.1)**:
- **Cross-stage caching**: Each PP stage caches blocks received from earlier stages locally. Only *incremental* new blocks are transmitted at stage transitions.
- With P physical stages and V virtual stages per stage, caching reduces peak per-transition communication from O(C) to O(P), enabling overlap with computation.

**Implementation challenges in TorchTitan**:
1. `PipelineStage` expects a single input/output tensor. Block AttnRes produces `(list[Tensor], Tensor)`.
   - **Solution**: Pack block representations into a single tensor for inter-stage transfer, unpack on receiving end. E.g., `torch.stack(blocks)` → send as one tensor of shape `[N, B, T, D]`.
2. The PP split logic (`pipeline_module_split`) deepcopies the model and deletes non-local layers. The `Decoder.forward()` must handle the case where only a subset of layers exists.
   - **Solution**: The existing Decoder already handles `None` modules for PP. The AttnRes decoder needs to accept `blocks` as input (from prior stage) and output `blocks` (to next stage).
3. Block boundaries may not align with PP stage boundaries (the paper notes this explicitly).
   - **Solution**: Track block accumulation state correctly. A block started in stage 0 may complete in stage 1.

**Cross-stage caching optimization** (for later, not MVP):
- Cache previously received blocks in each stage's local memory
- Only transmit new blocks at stage transitions
- This is the V× reduction described in the paper (Eq. 8)

**Verdict**: Significant engineering effort required. For MVP, can start without PP support and add it later.

### 4.4 Activation Checkpointing — MEDIUM RISK

**Current**: AC wraps each `TransformerBlock` with `checkpoint_wrapper`. During backward, the block's forward is re-executed to recompute activations.

**Impact**: With AttnRes, the `blocks` list is an input to each layer. Under AC:
- Block representations from earlier layers are inputs, not internally computed activations
- These inputs will be saved (they're function inputs, not intermediates)
- The AttnRes computation itself (stack, norm, einsum, softmax, einsum) adds minimal overhead

**Key concern**: The per-layer activation footprint. The paper states (§4.1):
> "The per-layer activation footprint remains identical to standard architectures, as activation checkpointing eliminates all inter-block attention intermediates, and the checkpointed input p_l matches the memory size of the hidden state h_l it replaces."

However, the `blocks` list grows as we go deeper. With selective AC, the blocks must be kept alive for all subsequent layers. With full AC, they need to be saved as inputs.

**Memory analysis**:
- Block AttnRes with N=8 blocks: stores 8 tensors of shape [B, T, D]
- This is 8× the memory of a single hidden state
- For typical configs (B=4, T=8192, D=4096, bf16): 8 × 4 × 8192 × 4096 × 2 bytes = 2 GB
- This is non-trivial but manageable

**Mitigation**:
- Selective AC will naturally save the block representations (they're inputs, not recomputed)
- Full AC: blocks are inputs to each checkpointed region, so they're saved at the boundary
- Consider: only checkpoint the AttnRes operation itself (it's cheap to recompute)

**Verdict**: Compatible but increases memory. Need to benchmark and potentially optimize.

### 4.5 Context Parallelism — MEDIUM RISK

**Current**: CP shards the sequence dimension across devices. It's applied to inner attention modules.

**Impact**: AttnRes operates on the full hidden state (same dimensions as the standard residual). The softmax attention over depth is per-token, so it's embarrassingly parallel across the sequence dimension.

**Key concern**: The `block_attn_res` function stacks and operates on tensors. If these tensors are sharded across CP workers, the operation needs to happen on the local shard consistently.

**Verdict**: Should work naturally since AttnRes is per-token on the depth dimension. No sequence-dimension communication is needed for the depth-wise attention.

### 4.6 torch.compile — LOW RISK

**Current**: Each `TransformerBlock` is compiled independently after AC wrapping.

**Impact**: The AttnRes operations (stack, norm, einsum, softmax) are standard PyTorch ops that compile well. The variable-length `blocks` list may cause graph breaks if the list length varies across calls.

**Key concern**: With Block AttnRes, the number of blocks grows as we go through layers (1 block at start, up to N blocks at the end). This means different layers have different `blocks` list lengths, which could cause recompilation.

**Mitigation**: Since each `TransformerBlock` is compiled separately and has a fixed number of input blocks (determined by its position), this should be fine — each compiled block has a static graph.

**Verdict**: Should work, but need to verify no graph breaks from list manipulation.

---

## 5. Phased Implementation Strategy

### MVP (Phase 1): No Parallelism -- COMPLETE
1. Implement `BlockAttnRes` module
2. Create `AttnResTransformerBlock` and `AttnResDecoder` in experiments folder
3. Create config registry with debug-size model configs
4. Test on single GPU with loss convergence
5. **Validation**: Compare loss curves with and without AttnRes using `--debug.seed=42 --debug.deterministic`

### Phase 2: FSDP + TP Support -- COMPLETE (GPU VERIFIED via fake_backend)
1. Created `parallelize.py` with TP plans for AttnRes modules
2. FSDP wrapping works out of the box (standard per-block `fully_shard`)
3. TP required solving three issues:
   - Element-wise projection instead of matmul to avoid view flatten errors
   - `distribute_tensor` for proj weights (parallelize_module only handles submodules)
   - `_ensure_dtensors()` to handle mixed DTensor/AsyncCollectiveTensor/Tensor types
4. All 4 fake_backend integration tests pass (FSDP, FSDP+TP, compile, FSDP+compile)

### Phase 3: AC + compile Support -- COMPLETE (GPU VERIFIED)
1. Selective AC verified -- gradient equivalence confirmed on CPU and GPU
2. torch.compile fullgraph verified -- no graph breaks with eager backend
3. AC reduces memory (0.67GiB vs 1.10GiB with FSDP+TP fake_backend)

### Phase 4: Pipeline Parallelism Support
1. Modify forward signature to pass `blocks` between stages
2. Implement block packing/unpacking for PP stage transfer
3. Implement cross-stage caching optimization
4. Test with various PP schedules (1F1B, interleaved)

### Phase 5: Verification & Comparison — IN PROGRESS
1. **Config alignment audit** — ✅ COMPLETE
   - debugmodel: aligned (audited)
   - 1B: gap fixed — added `enable_weight_tying=True`, implemented full weight
     tying support in AttnResDecoder (Config, __init__, init_weights, PP check,
     FSDP grouping), 7 tests verify correctness (47/47 total tests pass)
2. **Parallelism numerical verification**: Verify each parallelism config
   produces bitwise identical loss across repeated runs with
   `--debug.seed 42 --debug.deterministic`. Use TensorBoard for full precision.
   - FSDP: ✅ verified (2-GPU, 20 steps, all losses + grad_norms bitwise identical)
   - TP: ✅ verified (2-GPU, 20 steps, all losses + grad_norms bitwise identical)
   - FSDP+TP: ✅ verified (4-GPU, 20 steps, all losses + grad_norms bitwise identical)
3. **Baseline vs AttnRes comparison**: Run Llama3 and AttnRes with identical
   configs, compare across three dimensions (all logged by TorchTitan to TB):
   - **Convergence efficiency** (primary, paper's main claim):
     `loss_metrics/global_avg_loss` — AttnRes should reach lower loss at same
     step. Baseline should need ~1.25x more steps to reach same loss.
   - **Per-step overhead** (secondary, should be negligible):
     `throughput(tps)`, `mfu(%)` — expect <4% overhead per paper §4.1
   - **Memory overhead** (tertiary, small increase expected):
     `memory/max_active(GiB)` — AttnRes stores N block representations [B,T,D]
   - **Important**: The paper does NOT claim less compute/memory per step.
     AttnRes converges faster (fewer steps to same loss), not cheaper per step.
4. **Comparison at debugmodel scale** (500 steps): ✅ COMPLETE — Llama3 wins at
   this scale (AttnRes ahead steps 1–100, L3 overtakes at ~150, ends 0.22 loss
   ahead). TPS overhead 29% (expected to shrink at scale). Memory +1.6%.
5. **Comparison at 1B scale** (1000 steps): ✅ COMPLETE — AttnRes overtakes
   Llama3 at step 621, reaches 5.1% lower final loss. TPS overhead 36%.
   Memory overhead 0.2%. Directionally consistent with paper's claims.
6. Scaling law experiments (optional, requires significant compute)

---

## 6. File Structure

```
torchtitan/experiments/attn_residuals/
├── __init__.py
├── attn_res.py              # BlockAttnRes module
├── model.py                 # AttnResTransformerBlock, AttnResDecoder
├── config_registry.py       # Model configs (debug, small, medium)
├── parallelize.py           # TP/FSDP/AC parallelization
└── README.md                # Usage instructions
```

---

## 7. New Parameters Added per Layer

| Parameter | Shape | Count | Purpose |
|-----------|-------|-------|---------|
| `attn_res_proj.weight` | [1, d] | d | Pre-attention pseudo-query |
| `attn_res_norm.weight` | [d] | d | Pre-attention key normalization |
| `mlp_res_proj.weight` | [1, d] | d | Pre-MLP pseudo-query |
| `mlp_res_norm.weight` | [d] | d | Pre-MLP key normalization |
| **Total per layer** | | **4d** | |

For d=4096: **16,384 params/layer** — negligible vs. ~33M params/layer in a typical Llama3 block.

---

## 8. Critical Implementation Notes

1. **Zero-initialize pseudo-query vectors**: The paper emphasizes this is critical (§5). Initial alpha weights must be uniform across sources to prevent training volatility. `nn.init.zeros_(proj.weight)` achieves this since softmax of equal logits = uniform.

2. **RMSNorm on keys is essential**: Without it, layers with naturally larger magnitudes dominate the attention. The paper shows 0.007 loss degradation without RMSNorm (Table 4).

3. **Block size ~8 is optimal**: The paper sweeps block sizes and finds N≈8 blocks recovers most of full AttnRes gains (Figure 6). With 32 Llama3 layers, that's block_size=4 (4 layers per block, 8 sub-layers each = 8 blocks total).
   **FIXED (2026-04-02)**: The 8B config was set to `num_attn_res_blocks=16`.
   Changed to `8` (4 layers/block) to match the paper. Re-run completed — see
   REPORT.md for 3-way comparison results.

4. **Softmax over depth, not sigmoid**: The paper ablates this (Table 4) — softmax's competitive normalization is important for sharp selection among sources.

5. **Don't modify core torchtitan**: Per project rules, this goes in `torchtitan/experiments/`. No `if attn_res:` branches in core files.

6. **Forward signature change**: The `TransformerBlock.forward()` signature changes from `(x, freqs_cis, masks, positions) -> x` to `(blocks, partial_block, freqs_cis, masks, positions) -> (blocks, partial_block)`. This is the biggest architectural change and affects PP stage interfaces.

---

## 9. Open Questions

1. **Weight tying with AttnRes**: Does the AttnRes projection benefit from sharing across layers? The paper uses per-layer projections.

2. **MoE interaction**: For Llama4-style models with interleaved MoE layers, does AttnRes help more or less? The paper tests on MoE (Kimi Linear 48B) and shows gains.

3. **Checkpoint compatibility**: AttnRes models have different state dicts than base models. Need a state dict adapter for loading pretrained weights (with zero-initialized AttnRes params).

4. **Optimal block size for different model sizes**: The paper uses N≈8 for all sizes. May want to sweep for TorchTitan's debug/small configs.

6. **Block boundary ordering**: **FIXED (2026-04-02)**. Implementation now
   matches the paper: AttnRes BEFORE boundary check. Boundary resets
   `partial_block = None`. Decoder init changed from `zeros` to `None`.

7. **TPS overhead from unbatched computation**: **PARTIALLY FIXED (2026-04-02)**.
   The weighted sum loop `sum(w_i * v_i for ...)` was replaced with batched
   `(weights.unsqueeze(-1) * V).sum(dim=0)` where `V = torch.stack(sources)`.
   Per-source norm loop kept for TP safety. Cuts weighted-sum kernel launches
   from 2N to 3. All 47 tests pass including FSDP+TP. Full batching of the
   norm loop (via `F.rms_norm` on stacked tensor) deferred — needs TP testing
   of SequenceParallel placement on 4D tensors.

5. **Interaction with compile + AC + PP**: The triple combination needs careful testing. Start with simpler combinations first.
