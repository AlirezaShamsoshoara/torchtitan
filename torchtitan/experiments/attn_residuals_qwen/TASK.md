# Tasks: Attention Residuals for Qwen3 (MoE)

## Phase 1: Setup & Core Model

### Task 0: Create experiment folder structure
- **Status**: NOT STARTED
- **Description**: Create `attn_residuals_qwen/` under experiments with
  `__init__.py`, `model.py`, `parallelize.py`, `config_registry.py`, and
  `tests/` directory.
- **Acceptance**: Folder exists, empty `__init__.py` files in place.

### Task 1: Implement AttnResQwen3TransformerBlock
- **Status**: NOT STARTED
- **Description**: Extend `Qwen3TransformerBlock` with AttnRes parameters and
  forward logic. Must handle both dense and MoE sub-layers.
  - Add 4 parameters: `attn_res_proj`, `attn_res_norm`, `mlp_res_proj`,
    `mlp_res_norm`.
  - Compute `block_size` and `is_block_boundary` from `num_attn_res_blocks`.
  - Override `forward` to use `block_attn_res` from
    `attn_residuals/attn_res.py`, with MoE branching at step 5.
  - Implement `init_weights` with zero-init for proj weights.
- **Key decision**: Import `block_attn_res` from existing experiment (no dup).
- **Acceptance**: Block compiles, handles both `moe_enabled=True/False`,
  returns `(blocks, partial_block)`.

### Task 2: Implement AttnResQwen3Model (Decoder)
- **Status**: NOT STARTED
- **Description**: Extend `Qwen3Model` (which extends `Decoder`). Override
  `forward` to maintain `(blocks, partial_block)` state through the layer
  loop.
  - `blocks = [h]` after embedding (h = tok_embeddings output)
  - Each layer returns `(blocks, partial_block)`
  - Final output: `output(norm(partial_block))`
  - Preserve `update_from_config` from Qwen3Model (TP head validation,
    MoE force-load-balance propagation, etc.)
  - Preserve weight tying support.
- **Acceptance**: Decoder forward runs end-to-end, gradients flow, weight
  tying works when enabled.

## Phase 2: Configuration & Registration

### Task 3: Define model configs in `__init__.py`
- **Status**: NOT STARTED
- **Description**: Register AttnRes Qwen3 model configs.
  - `debugmodel`: Dense, dim=256, 8 layers, 4 blocks, vocab=2048,
    weight_tying=True. Matches Qwen3 debugmodel dims but with AttnRes.
  - `debugmodel_moe`: MoE, dim=256, 8 layers, 4 blocks, 64 experts, top-8,
    vocab=2048. Matches Qwen3 debugmodel_moe but with AttnRes.
  - `0.6B`: Dense, dim=1024, 28 layers, 8 blocks, vocab=151936,
    weight_tying=True. Matches Qwen3 0.6B.
  - `30B-A3B`: MoE, dim=2048, 48 layers, 8 blocks, 128 experts, top-8,
    vocab=151936. Matches Qwen3 30B-A3B.
  - All configs use `head_dim=128`, `rope_backend="cos_sin"`, `eps=1e-6`,
    QK normalization --- matching Qwen3 defaults.
- **Key decision**: Match Qwen3 hyperparameters exactly so the only
  difference is the residual connection mechanism.
- **Acceptance**: `model_registry(flavor)` returns valid `ModelSpec` for each
  config. Model builds on meta device without errors.

### Task 4: Register experiment in `experiments/__init__.py`
- **Status**: NOT STARTED
- **Description**: Add `"attn_residuals_qwen"` to `_supported_experiments`.
- **Acceptance**: `--module attn_residuals_qwen` is recognized by torchtitan.

### Task 5: Create trainer configs in `config_registry.py`
- **Status**: NOT STARTED
- **Description**: Define trainer config functions for running experiments.
  All Qwen3 baselines live in this experiment folder (not in core Qwen3).

  **Debug configs** (fast validation):
  - `attn_res_qwen3_debugmodel`: 10 steps, c4_test, lr=8e-4
  - `qwen3_debugmodel_baseline`: Matching Qwen3 baseline
  - `attn_res_qwen3_moe_debug`: 10 steps, c4_test, MoE, lr=8e-4
  - `qwen3_moe_debug_baseline`: Matching Qwen3 MoE baseline

  **Scale configs** (real evaluation):
  - `attn_res_qwen3_0_6b` / `qwen3_0_6b_baseline`: 1000 steps, full C4
  - `attn_res_qwen3_30b_a3b` / `qwen3_30b_a3b_baseline`: 5000 steps, full C4

- **Key decisions**:
  - Use same learning rates, batch sizes, and schedules for AttnRes and
    baseline (fair comparison).
  - Use `selective` activation checkpointing for debug, `full` for scale.
  - MoE configs set `ep_degree=1`, `etp_degree=1` for debug (single-GPU
    compatible); larger values for scale configs.
- **Acceptance**: All trainer configs load without errors via
  `--job.config.file <name>`.

## Phase 3: Parallelism

### Task 6: Implement `parallelize_attn_res_qwen3`
- **Status**: NOT STARTED
- **Description**: Create the parallelization function that composes Qwen3's
  existing parallelism with AttnRes-specific TP handling.

  **Strategy** (follow Qwen3's `parallelize_qwen3` order):
  1. Apply non-MoE TP (reuse Qwen3's `apply_non_moe_tp` or replicate its
     logic, then add AttnRes TP entries)
  2. Apply MoE EP/TP (reuse `apply_moe_ep_tp` from Llama4 --- unchanged)
  3. Apply CP
  4. Apply AC
  5. Apply compile (sparse-aware for MoE)
  6. Apply FSDP (reuse from Llama4 --- unchanged)

  **AttnRes-specific TP additions** (same pattern as Llama3 experiment):
  - `attn_res_norm`, `mlp_res_norm` -> `SequenceParallel()`
  - `attn_res_proj.weight`, `mlp_res_proj.weight` -> `distribute_tensor`
    with `[Replicate()]`

- **Key challenge**: Ensuring the TP plan additions compose correctly with
  Qwen3's existing QK norm SequenceParallel and MoE EP/ETP handling.
- **Acceptance**: FSDP, TP, FSDP+TP, FSDP+EP all run without errors on
  fake backend. Proj weights are Replicate DTensors, norms are
  SequenceParallel.

## Phase 4: Testing

### Task 7: Core AttnRes function tests (`test_attn_res.py`)
- **Status**: NOT STARTED
- **Description**: Test `block_attn_res` in Qwen3 context. Since the function
  is shared, focus on MoE-relevant scenarios:
  - Hidden states with Qwen3-like dimensions (dim=256, head_dim=128)
  - Verify function works with hidden states shaped as MoE output
  - Gradient flow through block_attn_res to MoE-scale tensors
  - Batched norm equivalence (can adapt from Llama3 tests)
- **Acceptance**: All tests pass on CPU.

### Task 8: Model unit tests (`test_model.py`)
- **Status**: NOT STARTED
- **Description**: Comprehensive block and decoder tests.

  **TestAttnResQwen3TransformerBlock**:
  - Output types (blocks list, partial_block tensor)
  - Block boundary detection
  - Blocks list immutability (no in-place mutation)
  - Zero-init projections produce uniform weights
  - Forward-backward (dense FFN)
  - Forward-backward (MoE FFN)
  - Param count: exactly 4*dim extra vs base Qwen3TransformerBlock
  - MoE routing still produces valid expert assignments with AttnRes

  **TestAttnResQwen3Model**:
  - Output shape
  - Forward-backward (dense)
  - Forward-backward (MoE)
  - All params have gradients
  - Weight tying (dense models)
  - ModelSpec correctness

- **Acceptance**: All tests pass on CPU. MoE-specific tests verify routing
  is unaffected by AttnRes.

### Task 9: Parallelism tests (`test_parallelize.py`)
- **Status**: NOT STARTED
- **Description**: TP plan and integration tests.

  **TestTPPlanCoverage**:
  - All TP plan entries correspond to actual submodules
  - AttnRes proj weights have correct shape [1, D]
  - AttnRes norms are separate from attention/FFN norms

  **TestCompileCompatibility**:
  - fullgraph=True with eager backend (dense block)
  - fullgraph=True with eager backend (MoE block)
  - Full decoder compilation

  **TestFakeBackendIntegration**:
  - FSDP (dense debugmodel)
  - FSDP (MoE debugmodel_moe)
  - FSDP + TP (dense)
  - FSDP + TP (MoE)
  - FSDP + EP (MoE only)

- **Acceptance**: All tests pass. Fake backend tests verify end-to-end
  training pipeline.

## Phase 5: Validation & Benchmarking

### Task 10: Debug model training validation (dense)
- **Status**: NOT STARTED
- **Description**: Run dense debugmodel end-to-end on real GPUs.
  - `torchrun --nproc_per_node=4 -m torchtitan.train --module attn_residuals_qwen --model.flavor debugmodel`
  - Verify loss decreases over 500 steps
  - Compare against Qwen3 debugmodel baseline
  - Verify deterministic with `--debug.seed=42 --debug.deterministic`
- **Acceptance**: Training completes, loss decreases monotonically (roughly).

### Task 11: Debug model training validation (MoE)
- **Status**: NOT STARTED
- **Description**: Run MoE debugmodel_moe end-to-end on real GPUs.
  - Same as Task 10 but with `--model.flavor debugmodel_moe`
  - Verify MoE routing works (experts get tokens)
  - Verify load balancing updates
  - Compare against Qwen3 MoE baseline
- **Acceptance**: Training completes, loss decreases, MoE routing is active.

### Task 12: FSDP/TP/EP determinism verification
- **Status**: NOT STARTED
- **Description**: Verify bitwise deterministic loss across parallelism
  configs with `--debug.seed=42 --debug.deterministic`.
  - FSDP only (dense and MoE)
  - TP only (dense)
  - FSDP + TP (dense and MoE)
  - FSDP + EP (MoE only)
- **Acceptance**: Identical loss across configurations for the same model.

### Task 13: TPS measurement and comparison
- **Status**: NOT STARTED
- **Description**: Measure tokens-per-second for AttnRes vs baseline Qwen3.
  - debugmodel (dense): AttnRes vs Qwen3 baseline
  - debugmodel_moe: AttnRes vs Qwen3 MoE baseline
  - Measure at 8-GPU scale
  - Report overhead percentage
  - Compare with Llama3 AttnRes overhead (17-28%)
- **Key question**: Does MoE routing overhead mask or compound with AttnRes
  depth-attention overhead?
- **Acceptance**: TPS numbers collected, overhead quantified.

### Task 14: Loss comparison at scale (0.6B dense)
- **Status**: NOT STARTED
- **Description**: Run 1000-step comparison on full C4 dataset.
  - AttnRes Qwen3 0.6B vs Qwen3 0.6B baseline
  - Same hyperparameters, batch size, learning rate
  - Generate loss comparison plots
- **Acceptance**: Loss curves generated and compared.

### Task 15: Loss comparison at scale (30B-A3B MoE)
- **Status**: NOT STARTED
- **Description**: Run 5000-step comparison on full C4 dataset.
  - AttnRes Qwen3 30B-A3B vs Qwen3 30B-A3B baseline
  - Requires 8x H100 GPUs with EP
  - Generate loss comparison and TPS comparison plots
- **Note**: This is the key experiment --- MoE at scale, matching the paper's
  architecture class.
- **Acceptance**: Loss curves and TPS numbers collected.

### Task 16: Memory overhead measurement
- **Status**: NOT STARTED
- **Description**: Measure peak GPU memory for AttnRes vs baseline.
  - debugmodel (dense), debugmodel_moe, 0.6B, 30B-A3B
  - Report overhead percentage
  - Compare with Llama3 experiment (was <2%)
- **Acceptance**: Memory numbers collected, overhead quantified.

## Phase 6: Pipeline Parallelism

### Task 17: Design block caching for PP
- **Status**: NOT STARTED
- **Description**: Design the block caching mechanism (paper Section 4.1) that
  makes AttnRes compatible with Pipeline Parallelism.

  **Problem**: PP splits the model across stages and passes activations via
  fixed-size `dist.isend/irecv`. AttnRes's growing `blocks` list violates
  three PP constraints:
  1. Variable-length list cannot be a fixed-size send/recv buffer
  2. Lists are not tensors (PipelineStage rejects non-tensor outputs)
  3. `partial_block` can be `None` at block boundaries

  **Approach** (paper's block caching, Option C):
  - Each stage receives a single `prev_block` tensor from the previous stage
    (same shape `[B, T, D]` as standard Llama3 PP --- fixed size)
  - Each stage maintains its own local `blocks` list internally, seeded with
    `prev_block` from the prior stage
  - Each stage outputs a single tensor to the next stage: its last completed
    block (or `partial_block` if no boundary was crossed)
  - Inter-stage communication is a single tensor per stage (identical to
    standard decoder PP)

  **Design deliverables**:
  - Document how blocks are partitioned across stages
  - Define the stage-local block accumulation logic
  - Specify how the first stage seeds `blocks = [embedding]`
  - Specify how the last stage produces the final output from `partial_block`
  - Analyze whether block caching changes model numerics vs non-PP AttnRes
    (it should be mathematically equivalent if each stage receives all prior
    blocks, but the caching optimization may approximate by only sending the
    most recent block --- verify this)
  - Determine if `pipeline_llm` / `pipeline_module_split` needs modification
    or if the model's forward can handle this internally

- **Dependencies**: Tasks 1-2 (core model must exist first)
- **Acceptance**: Design document with clear inter-stage protocol, block
  partitioning strategy, and numerical equivalence analysis.

### Task 18: Implement PP-compatible AttnRes forward
- **Status**: NOT STARTED
- **Description**: Modify `AttnResQwen3Model.forward` (or create a PP-specific
  variant) that supports pipeline parallelism via block caching.

  **Key changes**:
  - Forward accepts a single tensor input (tokens for first stage, hidden
    state for subsequent stages) --- same as standard `Decoder.forward`
  - Internally, non-first stages reconstruct `blocks` from the received
    tensor (treat it as the accumulated prior-stage block representation)
  - Each stage accumulates its own blocks locally during the layer loop
  - Output is a single tensor: the last completed block or `partial_block`
  - Handle `None` submodules (`tok_embeddings`, `norm`, `output`) for
    intermediate stages, matching standard PP tolerance pattern

  **Constraint**: Must NOT break non-PP forward path. Either use a flag to
  switch behavior or design the forward to work for both cases naturally.

- **Dependencies**: Task 17 (design), Task 6 (parallelism)
- **Acceptance**: Forward runs in PP mode with `pipeline_llm`. Single tensor
  passed between stages. Non-PP forward path still works identically.

### Task 19: PP parallelism integration
- **Status**: NOT STARTED
- **Description**: Update `parallelize_attn_res_qwen3` to support PP.
  - Register `pipeline_llm` as `pipelining_fn` in ModelSpec (already planned
    in Task 3, but verify it works end-to-end)
  - Ensure AttnRes TP additions compose with PP stage splitting
  - Verify that `pipeline_module_split` correctly splits AttnRes layers
    (the 4 extra parameters per block must stay with their layer)
  - Add PP-specific trainer configs with `pipeline_parallel_degree > 1`
- **Dependencies**: Task 18 (PP-compatible forward)
- **Acceptance**: PP runs end-to-end with fake backend and real GPUs.

### Task 20: PP testing and validation
- **Status**: NOT STARTED
- **Description**: Comprehensive PP testing.

  **Correctness tests**:
  - PP forward produces same final output shape as non-PP
  - Block caching produces numerically equivalent results to non-PP
    (if using full block passing) or document the approximation gap
    (if using paper's incremental caching)
  - PP + FSDP composition works
  - PP + TP composition works
  - PP + EP (MoE) composition works

  **Integration tests** (fake backend):
  - PP=2 with dense debugmodel
  - PP=2 with MoE debugmodel_moe
  - PP=2 + FSDP with dense
  - PP=2 + FSDP + TP with dense
  - PP=2 + FSDP + EP with MoE

  **GPU validation**:
  - Determinism: PP must produce bitwise identical loss to non-PP
    (if block caching is exact) or bounded divergence (if approximate)
  - TPS comparison: PP vs non-PP, targeting <4% overhead (paper's claim)
  - Memory: PP should reduce per-GPU memory (fewer layers per stage)

- **Dependencies**: Task 19 (PP integration)
- **Acceptance**: All PP tests pass. TPS overhead quantified. Numerical
  equivalence (or bounded divergence) documented.

### Task 21: PP TPS optimization (cross-stage caching)
- **Status**: NOT STARTED
- **Description**: Implement the paper's cross-stage caching optimization
  (Section 4.1) to minimize PP overhead.

  Without caching, each virtual stage transition sends all accumulated blocks
  to the next stage --- O(C) per transition, O(C^2) total. The paper's
  optimization:
  - Each physical stage caches blocks received during earlier virtual stages
  - Only the **incremental** block (newly completed in the current virtual
    stage) is transmitted at each stage transition
  - Per-transition cost drops from O(C) to O(P), a V-times improvement
  - Enables full overlap of block communication with computation

  This task is optional if Task 18's simpler single-tensor approach achieves
  acceptable overhead. Only needed if PP overhead exceeds ~10%.

- **Dependencies**: Task 20 (PP must be working and benchmarked first)
- **Acceptance**: PP TPS overhead reduced toward paper's <4% claim.

## Summary

| Phase | Tasks | Key Deliverable |
|-------|-------|----------------|
| 1. Core Model | 0-2 | Working AttnRes Qwen3 model (dense + MoE) |
| 2. Configuration | 3-5 | Model configs, trainer configs, experiment registration |
| 3. Parallelism | 6 | Parallelization with TP, EP, FSDP support |
| 4. Testing | 7-9 | Comprehensive test suite (CPU + fake backend) |
| 5. Validation | 10-16 | GPU training, determinism, TPS, loss, memory |
| 6. Pipeline Parallelism | 17-21 | PP with block caching, TPS optimization |
