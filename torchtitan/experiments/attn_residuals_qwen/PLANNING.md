# Planning: Attention Residuals for Qwen3 (MoE)

## Goal

Implement Block Attention Residuals for Qwen3, covering both dense and MoE
variants. Validate correctness, measure loss improvement vs baseline Qwen3,
and quantify TPS/memory overhead --- especially for MoE configurations.

## Design Decisions

### 1. Reuse `block_attn_res` from existing experiment

The core depth-attention function (`block_attn_res` in
`attn_residuals/attn_res.py`) is model-agnostic. It operates on hidden state
tensors regardless of whether they came from a dense FFN or MoE layer. We will
**import and reuse** this function rather than duplicating it.

**Rationale**: The function already handles DTensor/TP compatibility, batched
RMSNorm, and element-wise projection. No MoE-specific changes are needed.

### 2. Extend Qwen3TransformerBlock, not Llama3's AttnResTransformerBlock

We will create `AttnResQwen3TransformerBlock` by extending
`Qwen3TransformerBlock` (not `AttnResTransformerBlock` from the Llama3
experiment). This is because:

- Qwen3TransformerBlock has the MoE branching logic (`moe_enabled` flag)
- Qwen3TransformerBlock uses QK normalization and cos_sin RoPE config
- Extending Qwen3's block preserves all Qwen3-specific behavior
- We only need to add the 4 AttnRes parameters and override `forward`

The forward method follows the same pattern as the Llama3 AttnRes block:
1. `h = block_attn_res(blocks, partial_block, attn_res_proj, attn_res_norm)`
2. Block boundary handling
3. Attention sub-layer -> accumulate into `partial_block`
4. `h = block_attn_res(blocks, partial_block, mlp_res_proj, mlp_res_norm)`
5. FFN or MoE sub-layer -> accumulate into `partial_block`

The key difference from Llama3's version: step 5 branches on `moe_enabled`.

### 3. Extend Qwen3Model for the decoder

`AttnResQwen3Model` extends `Qwen3Model`. Override `forward` to maintain
`(blocks, partial_block)` state through the layer loop, matching the Llama3
AttnRes decoder pattern. Keep `update_from_config` (Qwen3's runtime
validation for TP head divisibility, MoE force-load-balance, etc.).

### 4. Parallelism: compose Qwen3's strategy with AttnRes additions

The parallelize function reuses Qwen3's existing parallelism strategy and
adds AttnRes-specific TP handling on top:

- **Non-MoE TP**: Reuse `apply_non_moe_tp` from Qwen3's parallelize.py, then
  add AttnRes norm (SequenceParallel) and proj weight (Replicate DTensor)
  distribution.
- **MoE EP/TP**: Reuse `apply_moe_ep_tp` from Llama4's parallelize.py (same
  function Qwen3 already uses). No AttnRes-specific changes needed --- MoE
  parallelism is internal to the MoE layer.
- **FSDP**: Reuse Qwen3's `apply_fsdp` (from Llama4). Per-block wrapping
  with MoE expert weight separation. Weight tying grouping if enabled.
- **AC, compile**: Reuse Qwen3's `apply_ac` and `apply_compile_sparse`.

AttnRes-specific additions to TP plan (same pattern as Llama3 experiment):
```python
# For each block:
layer.attn_res_norm -> SequenceParallel()
layer.mlp_res_norm  -> SequenceParallel()
layer.attn_res_proj.weight -> distribute_tensor(..., [Replicate()])
layer.mlp_res_proj.weight  -> distribute_tensor(..., [Replicate()])
```

### 5. Config strategy

**Model configs** (in `__init__.py`):

| Config | Type | dim | n_layers | blocks | MoE | Purpose |
|--------|------|-----|----------|--------|-----|---------|
| `debugmodel` | Dense | 256 | 8 | 4 | No | Fast CPU/fake-backend tests |
| `debugmodel_moe` | MoE | 256 | 8 | 4 | 64 experts, top-8 | MoE-specific testing |
| `0.6B` | Dense | 1024 | 28 | 8 | No | Small-scale dense comparison |
| `30B-A3B` | MoE | 2048 | 48 | 8 | 128 experts, top-8 | Full MoE evaluation |

**Trainer configs** (in `config_registry.py`):

- `attn_res_qwen3_debugmodel` -- 10 steps, c4_test
- `attn_res_qwen3_moe_debug` -- 10 steps, c4_test, MoE debug
- `qwen3_debugmodel_baseline` -- Qwen3 baseline for debug comparison
- `qwen3_moe_debug_baseline` -- Qwen3 MoE baseline for comparison
- `attn_res_qwen3_0_6b` -- 1000 steps, full C4, dense comparison
- `qwen3_0_6b_baseline` -- Qwen3 0.6B baseline
- `attn_res_qwen3_30b_a3b` -- 5000 steps, full C4, MoE comparison
- `qwen3_30b_a3b_baseline` -- Qwen3 30B-A3B baseline

All baseline configs live in this experiment folder (never modify core Qwen3).

### 6. Testing strategy

Three test files mirroring the Llama3 experiment structure:

1. **test_attn_res.py**: Core `block_attn_res` function tests. Can largely
   reuse the Llama3 tests since the function is shared. Add MoE-context
   tests verifying the function works with hidden states shaped as MoE
   layers produce them.

2. **test_model.py**: Block and decoder tests.
   - `TestAttnResQwen3TransformerBlock`: output types, block boundary,
     immutability, zero-init, forward-backward, param count, MoE forward,
     MoE+AttnRes forward-backward
   - `TestAttnResQwen3Model`: output shape, forward-backward, all params
     have grad, MoE model forward-backward, weight tying
   - `TestMoEInteraction`: verify AttnRes doesn't interfere with MoE
     routing, expert load balancing still works

3. **test_parallelize.py**: Parallelism correctness.
   - TP plan coverage (including MoE TP entries)
   - Fake backend integration (FSDP, FSDP+TP, FSDP+EP)
   - Compile compatibility
   - MoE-specific: EP + AttnRes interaction

## Architecture Diagram

```
Input tokens
    |
    v
[tok_embeddings]
    |
    v
blocks = [embedding],  partial_block = None
    |
    v
+---------------------------------------------------+
| AttnResQwen3TransformerBlock (repeated n_layers)   |
|                                                    |
|  1. h = block_attn_res(blocks, partial, proj, norm)|
|  2. if block_boundary: blocks.append(partial)      |
|  3. attn_out = attention(attn_norm(h))             |
|     partial = partial + attn_out                   |
|  4. h = block_attn_res(blocks, partial, proj, norm)|
|  5a. if moe: mlp_out = moe(ffn_norm(h))           |
|  5b. else:   mlp_out = ffn(ffn_norm(h))            |
|     partial = partial + mlp_out                    |
|  6. return (blocks, partial)                       |
+---------------------------------------------------+
    |
    v
output = linear(norm(partial_block))
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MoE routing interferes with AttnRes | Low | High | AttnRes operates on hidden states, not within MoE. Verify with unit tests. |
| EP breaks DTensor compatibility in block_attn_res | Low | High | EP is internal to MoE layer. AttnRes sees regular hidden states. Test with fake backend + EP. |
| TPS overhead compounds (AttnRes + MoE) | Medium | Medium | Measure separately. Paper claims <4% with PP; our Llama3 saw 17-28%. MoE routing adds more. |
| Batched RMSNorm incompatible with MoE hidden dims | Very Low | Low | block_attn_res normalizes over dim=-1 which is model dim, not expert dim. |
| Qwen3 weight tying + AttnRes + MoE | Low | Medium | MoE models don't use weight tying. Dense models do. Test both paths. |
| Large model (30B-A3B) runs need 8+ GPUs | Medium | Medium | Start with debugmodel_moe. Scale up only after debug validation. |

## Dependencies

- `torchtitan/experiments/attn_residuals/attn_res.py` -- import `block_attn_res`
- `torchtitan/models/qwen3/model.py` -- extend `Qwen3TransformerBlock`, `Qwen3Model`
- `torchtitan/models/qwen3/parallelize.py` -- reuse `apply_non_moe_tp` pattern
- `torchtitan/models/llama4/parallelize.py` -- reuse `apply_moe_ep_tp`, `apply_fsdp`
- `torchtitan/models/common/moe/` -- MoE components (used by Qwen3, not modified)

### 7. Pipeline Parallelism via block caching

PP is included as Phase 6 (after core model is validated without PP).

**The problem**: PP splits the model across stages and passes activations via
fixed-size `dist.isend/irecv`. AttnRes's growing `blocks` list violates three
PP constraints:
1. Variable-length list cannot be a fixed-size send/recv buffer
2. Lists are not tensors (`PipelineStage` rejects non-tensor outputs)
3. `partial_block` can be `None` at block boundaries

**The solution**: Block caching (paper Section 4.1). Each PP stage:
1. Receives a single `prev_block` tensor from the prior stage (fixed shape
   `[B, T, D]` --- same as standard decoder PP)
2. Seeds its local `blocks` list with `prev_block`
3. Runs its layers, accumulating blocks locally
4. Sends a single tensor (last completed block or `partial_block`) to the
   next stage

This reduces inter-stage communication to a single tensor per stage,
identical to standard Llama3/Qwen3 PP. The paper claims <4% TPS overhead.

**Why Phase 6** (not earlier):
- PP adds significant complexity (block partitioning across stages,
  numerical equivalence analysis, cross-stage caching optimization)
- Phases 1-5 validate AttnRes correctness and measure baseline overhead
  without PP. If AttnRes doesn't help Qwen3 at all, PP work is wasted.
- The Llama3 experiment also deferred PP and has useful lessons to draw from.

**Numerical equivalence**: Block caching where each stage receives ALL prior
blocks (packed into the inter-stage tensor) is mathematically equivalent to
non-PP. The paper's optimization (sending only the most recent block) may
introduce approximation --- this must be analyzed in Task 17.

**Architecture diagram with PP**:
```
Stage 0                    Stage 1                    Stage 2
-----------               -----------               -----------
tok_embeddings            (receives prev_block)     (receives prev_block)
layers 0-N                layers N+1-M              layers M+1-L, norm, output
   |                         |                         |
   v                         v                         v
local blocks=[emb]        local blocks=[prev]       local blocks=[prev]
partial_block=None        partial_block=None         partial_block=None
   |                         |                         |
   [run layers]              [run layers]              [run layers]
   |                         |                         |
   v                         v                         v
send: last block -------> seed blocks + run -------> seed blocks + run
   (single tensor)        send: last block           output(norm(partial))
                          (single tensor)
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MoE routing interferes with AttnRes | Low | High | AttnRes operates on hidden states, not within MoE. Verify with unit tests. |
| EP breaks DTensor compatibility in block_attn_res | Low | High | EP is internal to MoE layer. AttnRes sees regular hidden states. Test with fake backend + EP. |
| TPS overhead compounds (AttnRes + MoE) | Medium | Medium | Measure separately. Paper claims <4% with PP; our Llama3 saw 17-28%. MoE routing adds more. |
| Batched RMSNorm incompatible with MoE hidden dims | Very Low | Low | block_attn_res normalizes over dim=-1 which is model dim, not expert dim. |
| Qwen3 weight tying + AttnRes + MoE | Low | Medium | MoE models don't use weight tying. Dense models do. Test both paths. |
| Large model (30B-A3B) runs need 8+ GPUs | Medium | Medium | Start with debugmodel_moe. Scale up only after debug validation. |
| PP block caching approximation | Medium | High | Analyze numerical equivalence in Task 17 before implementing. Full block passing is exact; incremental caching may diverge. |
| PP + EP + AttnRes three-way composition | Medium | High | Test incrementally: PP alone, PP+FSDP, PP+TP, PP+EP. Each adds a dimension of complexity. |

## Dependencies

- `torchtitan/experiments/attn_residuals/attn_res.py` -- import `block_attn_res`
- `torchtitan/models/qwen3/model.py` -- extend `Qwen3TransformerBlock`, `Qwen3Model`
- `torchtitan/models/qwen3/parallelize.py` -- reuse `apply_non_moe_tp` pattern
- `torchtitan/models/llama4/parallelize.py` -- reuse `apply_moe_ep_tp`, `apply_fsdp`
- `torchtitan/models/common/moe/` -- MoE components (used by Qwen3, not modified)
- `torchtitan/distributed/pipeline_parallel.py` -- `pipeline_llm`, `pipeline_module_split` (Phase 6)

## Out of scope

- Inference optimization (two-phase inference from paper Section 4)
- Custom Triton kernels for depth attention
- Modifying any core torchtitan code or the existing attn_residuals experiment
