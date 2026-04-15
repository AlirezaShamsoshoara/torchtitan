# Summary: Attention Residuals for Qwen3 (MoE)

## One-liner

Implement Block Attention Residuals for Qwen3 (dense + MoE) to evaluate
depth-attention on MoE models and compare TPS/loss against dense Llama3 results.

## Current Status: PLANNING

All planning documents created. Implementation not started. Awaiting review.

## Task Progress

| Phase | Tasks | Status |
|-------|-------|--------|
| 1. Core Model | 0-2 | NOT STARTED |
| 2. Configuration | 3-5 | NOT STARTED |
| 3. Parallelism | 6 | NOT STARTED |
| 4. Testing | 7-9 | NOT STARTED |
| 5. Validation | 10-16 | NOT STARTED |
| 6. Pipeline Parallelism | 17-21 | NOT STARTED |

**Total**: 0/22 tasks complete

## Key Design Decisions

1. **Reuse `block_attn_res`** from existing Llama3 experiment (import, not
   copy). The function is model-agnostic.

2. **Extend Qwen3TransformerBlock** (not Llama3's AttnResTransformerBlock)
   to preserve MoE branching, QK normalization, and Qwen3-specific config.

3. **Extend Qwen3Model** for the decoder, overriding forward to maintain
   `(blocks, partial_block)` state.

4. **Compose parallelism**: Reuse Qwen3's TP + Llama4's EP/FSDP, add
   AttnRes-specific TP entries (norms as SequenceParallel, proj weights
   as Replicate).

5. **Match Qwen3 hyperparameters exactly** in all configs so the only
   variable is the residual connection mechanism.

6. **All baselines in experiment folder** --- never modify core Qwen3 code.

## What's Different from Llama3 AttnRes

| Aspect | Llama3 AttnRes | Qwen3 AttnRes |
|--------|---------------|---------------|
| FFN type | Dense only | Dense + MoE |
| Base model | Llama3TransformerBlock | Qwen3TransformerBlock |
| QK norm | No | Yes (Qwen3 feature) |
| RoPE | complex | cos_sin |
| EP support | N/A (no MoE) | Yes (MoE variants) |
| Weight tying | 1B only | 0.6B (dense) |
| Parallelism | FSDP, TP, FSDP+TP | FSDP, TP, EP, PP, FSDP+TP, FSDP+EP |
| PP support | Deferred | Phase 6 via block caching (paper Sec 4.1) |
| block_attn_res | Defined locally | Imported from Llama3 experiment |

## Expected Outcomes

- **MoE should benefit similarly to dense** at sufficient depth (28+ layers),
  since AttnRes is architecture-agnostic per the paper.
- **TPS overhead may compound** with MoE routing overhead, or may be masked
  if MoE routing dominates wall-clock time.
- **Memory overhead should remain negligible** (<2%), as AttnRes adds only
  O(N*d) block storage where N~8 blocks.
- **EP should not interfere** with AttnRes, as EP operates within the MoE
  layer while AttnRes operates at the layer/block level.
- **PP via block caching** (Phase 6) should reduce TPS overhead toward the
  paper's <4% claim. Each stage sends a single tensor (same as standard PP),
  avoiding the growing-blocks-list problem.

## Files

```
torchtitan/experiments/attn_residuals_qwen/
  README.md          -- Project overview and usage
  PLANNING.md        -- Design decisions and architecture
  TASK.md            -- Task breakdown with acceptance criteria
  REPORT.md          -- Results (to be populated)
  SUMMARY.md         -- This file
  __init__.py        -- (Task 3) Model registration
  model.py           -- (Tasks 1-2) AttnRes Qwen3 model
  parallelize.py     -- (Task 6) Parallelism
  config_registry.py -- (Task 5) Trainer configs
  tests/
    __init__.py
    test_attn_res.py     -- (Task 7) Core function tests
    test_model.py        -- (Task 8) Model unit tests
    test_parallelize.py  -- (Task 9) Parallelism tests
```
