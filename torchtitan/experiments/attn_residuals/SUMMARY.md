# Attention Residuals Implementation Progress

## Phase 1: Planning and Analysis (Complete)

- Read and analyzed the Attention Residuals paper (Kimi Team, Moonshot AI, 2025)
- Studied TorchTitan's architecture: `Decoder` → `TransformerBlock` hierarchy,
  `Configurable`/`Module` protocol, `ModelSpec`, parallelism infrastructure
- Identified all 5 model variants (llama3, llama4, qwen3, deepseek_v3, gpt_oss)
  share identical PreNorm residual pattern
- Assessed parallelism risks: FSDP (low), TP (medium), PP (high), AC (medium),
  CP (medium), compile (low)
- Created `PLANNING.md` with architecture analysis, risk assessment, and 5-phase
  implementation strategy
- Created `TASK.md` with 11 detailed tasks, verification (V-prefix) and testing
  (T-prefix) steps, dependency graph

## Phase 2: Core Implementation — Tasks 0–4 (Complete)

### Task 0: Experiment Scaffold
- Created `torchtitan/experiments/attn_residuals/` directory
- Set up `__init__.py` with `attn_res_configs` dict and `model_registry()` function
- Added `tests/__init__.py`
- Registered `attn_residuals` in `torchtitan/experiments/__init__.py` so it's
  discoverable via `--module attn_residuals`

### Task 1: Core `block_attn_res()` Function
- **File**: `attn_res.py`
- Implements depth-wise softmax attention over block representations
- Uses element-wise mul + sum (TP-safe) for pseudo-query projection
- RMSNorm on keys prevents magnitude-dominant layers from dominating attention
- `_ensure_dtensors()` helper handles mixed DTensor/AsyncCollectiveTensor/Tensor
  types under TP by converting sources to DTensors via `DTensor.from_local`
- 8 unit tests in `test_attn_res.py`

### Task 2: `AttnResTransformerBlock`
- **File**: `model.py`
- Extends `TransformerBlock` with 4 new parameter groups per layer:
  `attn_res_proj` [1,d], `attn_res_norm` [d], `mlp_res_proj` [1,d], `mlp_res_norm` [d]
- Uses TorchTitan's `Linear` (Module-protocol-compatible) for proj weights
- Block boundary detection via `layer_id % block_size == 0`
- Immutable blocks list (`blocks + [partial_block]`) to avoid AC mutation bugs
- Forward signature: `(blocks, partial_block, freqs_cis, masks, positions)` ->
  `(blocks, partial_block)`
- Zero-initialized pseudo-query projections (paper Section 5)

### Task 3: `AttnResDecoder`
- **File**: `model.py`
- Overrides `Decoder.forward()` to maintain blocks list and partial_block
- Token embedding serves as initial block (b_0 in paper)
- `partial_block` starts as zeros, accumulates sub-layer outputs
- Handles `None` modules for Pipeline Parallelism compatibility
- Config includes `update_from_config` and `get_nparams_and_flops`

### Task 4: Config and Model Spec
- **File**: `__init__.py` -- Model configs (`debugmodel`: dim=256/6 layers/3 blocks,
  `1B`: dim=2048/16 layers/8 blocks, `8B`: dim=4096/32 layers/16 blocks) and
  `model_registry()` returning `ModelSpec`
- **File**: `config_registry.py` -- Trainer config presets: `attn_res_debugmodel()`,
  `attn_res_1b()`, `llama3_1b_baseline()`, `attn_res_1b_c4()`, `llama3_1b_baseline_c4()`,
  `attn_res_8b()`, `llama3_8b_baseline()`

## Phase 3: Low-Risk Tasks -- 6 (AC), 8 (compile), 11 (lint) (Complete)

### Task 6: Activation Checkpointing (Verified)
- Added `test_full_ac_preserves_gradients` (block-level): wraps a single
  `AttnResTransformerBlock` with `ptd_checkpoint_wrapper` and verifies gradients
  are bitwise identical to the no-AC baseline.
- Added `test_full_ac_decoder_gradients` (decoder-level): applies full AC to all
  layers in the decoder and verifies gradient equivalence across all parameters.
- **Result**: AC is fully compatible. Verified on both CPU and GPU (fake_backend).
  AC reduces memory (0.67GiB vs 1.10GiB with FSDP+TP).

### Task 8: torch.compile (Verified)
- Added `test_compile_eager_backend` (block-level): compiles with
  `torch.compile(backend="eager", fullgraph=True)`, forward + backward pass.
- Added `test_compile_eager_decoder` (decoder-level): compiles each layer,
  verifies output shape, no NaN, and gradient flow.
- **Result**: All operations (`torch.stack`, `RMSNorm`, element-wise mul+sum,
  `softmax`) are fully compilable with `fullgraph=True`.

### Task 11: Lint and Pre-commit (Complete)
- `ruff check` and `ruff format --check` pass with no issues on all 9 files.

## Phase 4: Parallelization -- Tasks 5, 8 Extended (Complete)

### Task 5: FSDP + TP (GPU Verified via fake_backend)
- **File**: `parallelize.py`
  - TP plan covers 15 module entries: standard attention/FFN + AttnRes norms
  - Proj weights distributed as `Replicate` DTensors via `distribute_tensor`
    (not via `parallelize_module` which only handles submodules)
  - `_ensure_dtensors()` in `attn_res.py` handles mixed DTensor/Tensor types
    that arise from TP's AsyncCollectiveTensor outputs
- **File**: `tests/test_parallelize.py` -- 13 tests total
- **TP plan coverage tests** (3 tests):
  - `test_tp_plan_keys_match_submodules`: verifies all 15 TP plan entries
    correspond to actual submodules in `AttnResTransformerBlock`
  - `test_attn_res_proj_in_tp_plan_as_replicate`: verifies proj weights [1,D]
    have correct shape (distributed separately via `distribute_tensor`)
  - `test_attn_res_norms_are_separate_from_standard_norms`: verifies AttnRes norms
    are distinct modules from attention_norm/ffn_norm
- **Parallelize function validation** (2 tests):
  - `test_signature_parameters`: verifies `parallelize_attn_res` accepts all
    required keyword arguments (model, parallel_dims, training, etc.)
  - `test_model_spec_parallelize_fn_is_parallelize_attn_res`: verifies ModelSpec
    correctly wires the parallelize function
- **GPU integration tests** (4 tests, all passing with CUDA):
  - `test_fsdp_fake_backend`: FSDP with fake distributed backend ✅
  - `test_fsdp_tp_fake_backend`: FSDP + TP with fake backend ✅
  - `test_compile_fake_backend`: torch.compile with fake backend ✅
  - `test_fsdp_compile_fake_backend`: FSDP + compile with fake backend ✅

### Task 8: torch.compile Extended (Verified)
- **Additional compile tests** in `test_parallelize.py` (4 tests):
  - `test_compile_block_fullgraph_eager`: single block compiles with fullgraph
  - `test_compile_different_block_positions`: blocks at different positions
    (different `blocks` list sizes) compile independently
  - `test_compile_decoder_all_layers`: full decoder with all layers compiled
  - `test_compile_numerics_match_eager`: compiled output matches eager output

### Key TP Challenges Solved

1. **DTensor view flatten error**: `aten.view.default` fails when matmul tries
   to reshape `[B, T, D]` to `[B*T, D]` with T (dim 1) sharded. Fixed by using
   element-wise `(norm(v) * w).sum(dim=-1)` instead of matmul.

2. **Mixed DTensor/Tensor error**: TP outputs (AsyncCollectiveTensor) lose DTensor
   metadata when stored in Python lists. `_ensure_dtensors()` converts plain
   tensors to DTensors via `DTensor.from_local(v, mesh, [Shard(1)])`.

3. **parallelize_module parameter limitation**: `parallelize_module` only handles
   submodules, not individual parameters. Proj weights are distributed separately
   via `distribute_tensor(proj.weight, tp_mesh, [Replicate()])`.

### Infrastructure
- Registered `attn_residuals` in `torchtitan/experiments/__init__.py`
  (`_supported_experiments` frozenset) so training can be launched via:
  `--module attn_residuals --config attn_res_debugmodel`

## Test Results

All 47 tests pass (with CUDA available):

```
test_attn_res.py:       8 passed
test_model.py:         26 passed (19 original + 7 weight tying)
test_parallelize.py:   13 passed (incl. 4 GPU integration tests)
Total:                 47 passed, 0 failed
```

Lint clean: `ruff check` and `ruff format --check` pass with no issues.

## Phase 5: Verification & Comparison — COMPLETE

### 5.1 Config Alignment Audit — COMPLETE

- **debugmodel**: All architecture and training params match between Llama3 and AttnRes.
- **1B**: Gap fixed — added `enable_weight_tying=True`, implemented full weight
  tying support in `AttnResDecoder` (7 tests verify correctness).
- **1B trainer configs**: Created `attn_res_1b()` and `llama3_1b_baseline()` in
  `config_registry.py`. Both share identical training hyperparameters (lr=3e-4,
  seq_len=4096, batch=2, selective AC). The Llama3 baseline config lives in the
  AttnRes experiment to avoid modifying core code.
- **Full C4 configs**: Added `attn_res_1b_c4()` and `llama3_1b_baseline_c4()` for
  streaming from the full C4 dataset. Blocked by corporate proxy filtering
  `cas-bridge.xethub.hf.co` — ready for use when network access is available.

### 5.2 Parallelism Numerical Verification — COMPLETE

All three parallelism configs produce **bitwise identical** loss and grad_norm:
- FSDP (2-GPU): ✅ verified
- TP (2-GPU): ✅ verified
- FSDP+TP (4-GPU): ✅ verified

### 5.3 AttnRes vs Llama3 Comparison — COMPLETE

**debugmodel** (500 steps, 1 GPU): Llama3 wins at this tiny scale. AttnRes
ahead for steps 1–100, Llama3 overtakes at ~150. Too few layers/blocks for
depth-selective attention to help. TPS overhead 29%, memory +1.6%.

**1B on c4_test** (1000 steps, 8 GPUs FSDP): AttnRes 5.1% lower avg loss
(last 50 steps). Both models memorize the tiny dataset (loss < 0.05).
AttnRes better in 66% of steps 800–1000. TPS overhead 36%, memory +0.2%.

**1B on full C4** (1000 steps, 8 GPUs FSDP): **AttnRes consistently lower
from step ~100** (99-100% of steps). 1.0% lower avg loss (last 50 steps).
Genuine generalization signal — neither model memorizes. Cleaner result
than c4_test. TPS overhead 36%, memory +0.2%.

See [REPORT.md](REPORT.md) for full results, reproduction commands, and analysis.

Loss plots: `loss_debugmodel_c4test.png`, `loss_1b_c4test.png`,
`loss_1b_c4.png`, `loss_1b_c4test_vs_c4.png`, `loss_comparison_combined.png`.

### Task 12 Status

| Step | Description | Status |
|------|-------------|--------|
| 12.1 | Fix `enable_weight_tying` gap in AttnRes 1B config | ✅ Complete |
| 12.1 | Verify AttnResDecoder supports weight tying (7 tests) | ✅ Complete |
| 12.2a | FSDP determinism verified (bitwise identical across runs) | ✅ Complete |
| 12.2b | TP determinism verified (bitwise identical across runs) | ✅ Complete |
| 12.2c | FSDP+TP determinism verified (bitwise identical across runs) | ✅ Complete |
| 12.3a | debugmodel: Llama3 vs AttnRes 500-step comparison | ✅ Complete |
| 12.3b | Compare loss curves (convergence efficiency) | ✅ Complete (L3 wins at debugmodel scale) |
| 12.3c | Compare TPS/MFU (per-step overhead, expect <4%) | ✅ Complete (29% overhead at small scale) |
| 12.3d | Compare peak memory (small increase expected) | ✅ Complete (1.6% overhead) |
| 12.3e | Compute steps-to-target-loss ratio (expect ~1.25x) | ✅ Complete (N/A — L3 better at this scale) |
| 12.4a | 1B: Llama3 vs AttnRes on c4_test (8 GPUs, 1000 steps) | ✅ Complete |
| 12.4b | 1B: Llama3 vs AttnRes on full C4 (8 GPUs, 1000 steps) | ✅ Complete (1.0% lower avg loss) |

## Phase 6: 8B Scale Verification — COMPLETE (Issues Found)

The 1B results show AttnRes works but the paper's main claims (1.25x compute
equivalence, <4% TPS overhead) are at 7B+ scale. Phase 6 runs the comparison
at Llama3 8B scale (dim=4096, 32 layers) with more training steps to fully
verify the paper's headline numbers.

### Tasks 13.1-13.3: Config and Infrastructure (Complete)

- **Task 13.1**: Added `8B` model config to `__init__.py` — dim=4096, 32 layers,
  16 blocks (**BUG: should be 8 per paper**), n_heads=32, n_kv_heads=8,
  ffn_dim=14336. All architecture params verified identical to Llama3 8B.
- **Task 13.2**: Added `attn_res_8b()` and `llama3_8b_baseline()` trainer configs
  to `config_registry.py`. Shared `_8b_trainer_config()` with lr=3e-4, batch=1,
  seq_len=8192, 5000 steps, full C4, selective AC, checkpoint every 1000 steps.
- **Task 13.3**: Added task `"13"` to `verify_parallelism.py` with
  `LLAMA3_8B_COMMON_ARGS`, `ATTNRES_8B_COMMON_ARGS`, and 8-GPU FSDP config.
  Extended milestone steps to include 1000-5000 for longer runs.

### Tasks 13.4-13.8: 8B Runs (Complete — AttnRes Regressed)

- **Task 13.4-13.5**: Both models ran successfully for 5000 steps on full C4.
  - Llama3 8B final loss: 3.6943
  - AttnRes 8B final loss: 3.8645 (**4.7% worse**)
- **Task 13.6**: Steps-to-target-loss: N/A — AttnRes never catches Llama3.
  Llama3 lower in 98.4% of all steps.
- **Task 13.7**: TPS overhead: 42.7% (paper claims <4%). Memory: 0.2% (matches paper).
- **Task 13.8**: Loss plots generated (`loss_8b_comparison_5000steps.png`),
  REPORT.md updated with full results and cross-scale summary.

### Issues Found

1. **Wrong block count (HIGH)**: 8B used `num_attn_res_blocks=16` (2 layers/block).
   Paper recommends 8 (4 layers/block). The 1B config correctly uses 8 blocks
   and shows improvement. Doubling to 16 doubles overhead and changes quality.
2. **Boundary ordering (MEDIUM)**: Implementation resets `partial_block` to zeros
   BEFORE the AttnRes call. PLANNING pseudocode does AttnRes FIRST — the first
   AttnRes in each new block sees zeros instead of meaningful content.
3. **No final aggregation (LOW-MEDIUM)**: Decoder uses `partial_block` directly;
   a final AttnRes call could improve output quality.

### 8B Plan

| Parameter | Llama3 8B | AttnRes 8B |
|-----------|-----------|------------|
| dim | 4096 | 4096 |
| n_layers | 32 | 32 |
| num_blocks | — | 16 (2 layers/block) **BUG: should be 8** |
| vocab_size | 128,256 | 128,256 |
| n_heads | 32 | 32 |
| n_kv_heads | 8 | 8 |
| params | ~8.03B | ~8.03B + 0.5M (negligible) |

### Training Plan

| Parameter | Value |
|-----------|-------|
| GPUs | 8x H100 (single node) |
| Parallelism | FSDP dp_shard=8 |
| lr | 3e-4 |
| local_batch_size | 1 |
| seq_len | 8192 |
| steps | 5,000–10,000 |
| dataset | c4 (full, streamed) |
| AC | selective |
| seed | 42, deterministic |

### What to Measure

1. **Steps-to-target-loss ratio**: The paper's main claim. For each target
   loss, compare how many steps each model needs. Expect Llama3 to need
   ~1.25x more steps.
2. **TPS overhead**: Expect <4% at 8B scale (vs 36% at 1B).
3. **Loss curves**: AttnRes should be consistently lower, with the gap
   widening over more training steps.
4. **Memory overhead**: Expect <1% (16 blocks at [1, 8192, 4096] bf16 =
   1 GB, negligible vs ~18 GiB working set).

### Task 13 Status

| Step | Description | Status |
|------|-------------|--------|
| 13.1 | Create AttnRes 8B model config | ✅ Complete |
| 13.2 | Create 8B trainer configs (attn_res_8b, llama3_8b_baseline) | ✅ Complete |
| 13.3 | Add 8B task to verify_parallelism.py (task 13) | ✅ Complete |
| 13.4 | Run Llama3 8B baseline (5000 steps, full C4) | ✅ Complete (loss: 3.6943) |
| 13.5 | Run AttnRes 8B (5000 steps, full C4) | ✅ Complete (loss: 3.8645, 4.7% worse) |
| 13.6 | Compute steps-to-target-loss ratio | ✅ N/A — AttnRes never catches up |
| 13.7 | Compare TPS overhead | ✅ 42.7% (bug: 16 blocks instead of 8) |
| 13.8 | Generate loss plots and update REPORT.md | ✅ Complete |

## All Tasks Status

| Task | Description | Status | Risk |
|------|-------------|--------|------|
| 0-4 | Core implementation | ✅ Complete | Done |
| 5 | FSDP + TP | ✅ Complete (fake_backend verified) | Done |
| 6 | AC | ✅ Complete (CPU + GPU verified) | Done |
| 7 | Pipeline Parallelism support | Deferred | High |
| 8 | torch.compile | ✅ Complete (eager + fake_backend) | Done |
| 9 | Numerical verification campaign | ✅ Complete (FSDP, TP, FSDP+TP determinism verified) | Done |
| 10 | Comprehensive test suite | 47/47 tests passing | Done |
| 11 | Lint | ✅ Complete | Done |
| 12 | AttnRes vs Llama3 comparison (1B) | ✅ Complete — c4_test + full C4 | Done |
| 13 | AttnRes vs Llama3 comparison (8B) | ✅ First run complete — AttnRes 4.7% worse (wrong block count). Issues documented. | Medium |

### How to Run All Tests

```bash
python -m pytest torchtitan/experiments/attn_residuals/tests/ -v
```

### Multi-GPU Training Commands

```bash
# FSDP only (2 GPUs)
NGPU=2 LOCAL_RANK=0 python -m torchtitan.train \
    --module attn_residuals --config attn_res_debugmodel \
    --comm.mode=fake_backend --training.steps 1 \
    --parallelism.data_parallel_shard_degree 2

# FSDP + TP (4 GPUs)
NGPU=4 LOCAL_RANK=0 python -m torchtitan.train \
    --module attn_residuals --config attn_res_debugmodel \
    --comm.mode=fake_backend --training.steps 1 \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2
```
