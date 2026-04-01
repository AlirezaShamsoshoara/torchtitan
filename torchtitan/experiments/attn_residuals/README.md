# Attention Residuals for TorchTitan

This experiment implements **Block Attention Residuals (AttnRes)** from the paper
*"Attention Residuals"* by the Kimi Team (Moonshot AI, 2025). AttnRes replaces
fixed unit-weight residual connections with learned, input-dependent softmax
attention over depth, yielding better loss at the same compute budget.

## Background

Standard Transformer residual connections accumulate all layer outputs with
fixed unit weights: `h_l = h_{l-1} + f_{l-1}(h_{l-1})`. This causes PreNorm
dilution (hidden-state magnitudes grow unboundedly with depth), irreversible
information loss (early layer outputs cannot be selectively recovered), and
output growth instability.

AttnRes addresses this by replacing the fixed accumulation with:

```
h_l = sum_{i} alpha_{i->l} * v_i
```

where `alpha_{i->l}` are softmax attention weights computed from a learned
pseudo-query vector per layer. **Block AttnRes** groups layers into N blocks
(~8), applying standard residual accumulation within each block and softmax
attention across block-level representations. This reduces memory from O(Ld) to
O(Nd) while recovering most of the full AttnRes gains.

Key results from the paper:
- Block AttnRes matches the loss of a baseline trained with **1.25x more compute**
- Training overhead: <4% with pipeline parallelism, negligible without
- Inference overhead: <2%
- Improvements across all benchmarks (MMLU +1.1, GPQA-Diamond +7.5, HumanEval +3.1)

## File Structure

```
torchtitan/experiments/attn_residuals/
├── __init__.py            # Model configs + model_registry()
├── attn_res.py            # Core block_attn_res() + _ensure_dtensors()
├── model.py               # AttnResTransformerBlock + AttnResDecoder
├── config_registry.py     # Trainer config presets
├── parallelize.py         # TP/FSDP/AC/compile parallelization
├── tests/
│   ├── test_attn_res.py       # Unit tests for core module (8 tests)
│   ├── test_model.py          # Unit tests for model (26 tests incl. weight tying)
│   ├── test_parallelize.py    # TP/compile/integration tests (13 tests)
│   └── verify_parallelism.py  # GPU parallelism verification script
├── loss_debugmodel_c4test.png  # Loss plot: debugmodel comparison
├── loss_1b_c4test.png         # Loss plot: 1B on c4_test
├── loss_1b_c4.png             # Loss plot: 1B on full C4
├── loss_1b_c4test_vs_c4.png   # Loss plot: c4_test vs C4 side-by-side
├── loss_comparison_combined.png # Combined 3-panel loss plot
├── PLANNING.md            # Architecture analysis + risk assessment
├── TASK.md                # Detailed implementation task breakdown
├── SUMMARY.md             # Progress tracking
├── REPORT.md              # Verification & comparison results
└── README.md              # This file
```

## How to Run

### Prerequisites

All commands use the `titan` conda environment:

```bash
conda activate titan
```

### Run Unit Tests

```bash
# All 47 tests
python -m pytest torchtitan/experiments/attn_residuals/tests/ -v

# Core AttnRes function tests only
python -m pytest torchtitan/experiments/attn_residuals/tests/test_attn_res.py -v

# Model tests only
python -m pytest torchtitan/experiments/attn_residuals/tests/test_model.py -v

# Parallelization and compile tests
python -m pytest torchtitan/experiments/attn_residuals/tests/test_parallelize.py -v
```

### Run Lint

```bash
ruff check torchtitan/experiments/attn_residuals/
ruff format --check torchtitan/experiments/attn_residuals/
```

### Single-GPU Training (requires GPU)

```bash
python -m torchtitan.train \
    --module attn_residuals \
    --config attn_res_debugmodel \
    --training.steps 100 \
    --debug.seed 42 \
    --debug.deterministic
```

### Dry-Run with Fake Backend (validates parallelization without real multi-GPU)

```bash
# FSDP only
NGPU=2 LOCAL_RANK=0 python -m torchtitan.train \
    --module attn_residuals --config attn_res_debugmodel \
    --comm.mode=fake_backend --training.steps 1 \
    --parallelism.data_parallel_shard_degree 2

# FSDP + TP
NGPU=4 LOCAL_RANK=0 python -m torchtitan.train \
    --module attn_residuals --config attn_res_debugmodel \
    --comm.mode=fake_backend --training.steps 1 \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2
```

### Multi-GPU Training with FSDP (requires 2+ GPUs)

```bash
torchrun --nproc_per_node=2 -m torchtitan.train \
    --module attn_residuals \
    --config attn_res_debugmodel \
    --parallelism.data_parallel_shard_degree 2
```

### Multi-GPU Training with FSDP + TP (requires 4+ GPUs)

```bash
torchrun --nproc_per_node=4 -m torchtitan.train \
    --module attn_residuals \
    --config attn_res_debugmodel \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2
```

### Multi-GPU with torch.compile (requires 2+ GPUs)

```bash
torchrun --nproc_per_node=2 -m torchtitan.train \
    --module attn_residuals \
    --config attn_res_debugmodel \
    --compile.enable
```

### Parallelism Numerical Verification (requires multi-GPU)

```bash
# Run all parallelism verification (FSDP, TP, FSDP+TP determinism checks)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py

# Run a specific task
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.2a

# See REPORT.md for detailed results
```

### Model Comparison (requires GPUs)

```bash
# debugmodel: Llama3 vs AttnRes (1 GPU, 500 steps)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.3a

# 1B: Llama3 vs AttnRes on c4_test (8 GPUs, 1000 steps)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.4a

# 1B: Llama3 vs AttnRes on full C4 (8 GPUs, 1000 steps, requires HF access)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.4b

# 8B: Llama3 vs AttnRes on full C4 (8 GPUs, 5000 steps, requires HF access)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 13

# Run everything (determinism + all comparisons)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task all
```

Loss plots are generated automatically and saved to the output directory.

## Available Model Configs

| Config | dim | n_layers | num_blocks | vocab_size | Use Case |
|--------|-----|----------|------------|------------|----------|
| `debugmodel` | 256 | 6 | 3 | 2,048 | Testing and development |
| `1B` | 2048 | 16 | 8 | 128,256 | Small-scale training |
| `8B` | 4096 | 32 | 16 | 128,256 | Paper-scale verification |

## Available Trainer Configs

| Config | Model | dataset | seq_len | batch | lr | GPUs | Use Case |
|--------|-------|---------|---------|-------|----|------|----------|
| `attn_res_debugmodel` | AttnRes debug | c4_test | 2048 | 8 | 8e-4 | 1+ | Testing |
| `attn_res_1b` | AttnRes 1B | c4_test | 4096 | 2 | 3e-4 | 8 | 1B comparison |
| `llama3_1b_baseline` | Llama3 1B | c4_test | 4096 | 2 | 3e-4 | 8 | 1B baseline |
| `attn_res_1b_c4` | AttnRes 1B | c4 (full) | 4096 | 2 | 3e-4 | 8 | 1B on full C4 |
| `llama3_1b_baseline_c4` | Llama3 1B | c4 (full) | 4096 | 2 | 3e-4 | 8 | 1B baseline on full C4 |
| `attn_res_8b` | AttnRes 8B | c4 (full) | 8192 | 1 | 3e-4 | 8 | 8B paper-scale |
| `llama3_8b_baseline` | Llama3 8B | c4 (full) | 8192 | 1 | 3e-4 | 8 | 8B baseline |

The `llama3_1b_baseline` / `llama3_1b_baseline_c4` configs live in the AttnRes
experiment to avoid modifying core Llama3 code. They use identical training
hyperparameters to their AttnRes counterparts for a fair comparison.

**Note**: Full C4 streaming requires network access to `cas-bridge.xethub.hf.co`
(HuggingFace data CDN). If blocked by a corporate proxy/firewall, use the
`c4_test` variants instead.

## Architecture Details

### New Parameters per Layer

Each `AttnResTransformerBlock` adds 4 small parameter groups beyond a standard
Transformer block:

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| `attn_res_proj.weight` | [1, d] | Pre-attention pseudo-query |
| `attn_res_norm.weight` | [d] | Pre-attention key normalization |
| `mlp_res_proj.weight` | [1, d] | Pre-MLP pseudo-query |
| `mlp_res_norm.weight` | [d] | Pre-MLP key normalization |

Total: **4d parameters per layer** (e.g., 16,384 for d=4096 -- negligible vs.
~33M params in a standard Llama3 block).

### Critical Design Choices

1. **Zero-initialized pseudo-query projections**: Ensures uniform initial
   attention weights across sources, preventing training volatility (paper Section 5).

2. **RMSNorm on keys**: Prevents layers with naturally larger magnitudes from
   dominating the attention weights. The paper shows 0.007 loss degradation
   without it.

3. **Softmax (not sigmoid) over depth**: Competitive normalization produces
   sharper selection among sources.

4. **Immutable blocks list**: Uses `blocks + [partial_block]` (list
   concatenation) instead of `blocks.append()` to avoid mutation bugs with
   activation checkpointing recomputation.

5. **Element-wise projection**: Uses `(norm(v) * w).sum(dim=-1)` instead of
   matmul to avoid `aten.view` flatten errors when the sequence dimension is
   sharded under TP.

6. **DTensor type handling**: `_ensure_dtensors()` converts plain tensors
   (from AsyncCollectiveTensor resolution) to DTensors for the weighted sum,
   preventing mixed Tensor/DTensor errors under TP.

### Parallelism Support

| Parallelism | Status | Notes |
|-------------|--------|-------|
| FSDP | Verified | Standard per-block wrapping, fake_backend tested |
| TP | Verified | AttnRes norms use SequenceParallel, proj weights Replicate DTensors |
| AC | Verified | Full and selective AC, gradients match no-AC baseline |
| torch.compile | Verified | fullgraph=True, eager backend + fake_backend tested |
| PP | Deferred | Highest risk -- needs block packing for inter-stage transfer |
| CP | Untested | Should work (AttnRes is per-token on depth dim) |

## Current Status

See [SUMMARY.md](SUMMARY.md) for detailed progress tracking.

### Task Status

| Task | Description | Status |
|------|-------------|--------|
| 0 | Experiment scaffold | ✅ Complete |
| 1 | Core `block_attn_res()` function | ✅ Complete |
| 2 | `AttnResTransformerBlock` | ✅ Complete |
| 3 | `AttnResDecoder` | ✅ Complete |
| 4 | Config and Model Spec | ✅ Complete |
| 5 | FSDP + TP parallelization | ✅ Complete (fake_backend verified) |
| 6 | Activation Checkpointing | ✅ Complete (CPU + GPU verified) |
| 7 | Pipeline Parallelism | Deferred (highest risk) |
| 8 | torch.compile | ✅ Complete (eager + fake_backend verified) |
| 9 | Numerical verification | ✅ Complete (FSDP, TP, FSDP+TP determinism verified) |
| 10 | Comprehensive test suite | ✅ Complete (47/47 tests passing) |
| 11 | Lint and pre-commit | ✅ Complete |
| 12 | AttnRes vs Llama3 comparison (1B) | ✅ Complete (c4_test: 5.1%, C4: 1.0% lower loss) |
| 13 | AttnRes vs Llama3 comparison (8B) | In progress — configs ready, GPU runs pending |

## Next Steps: 8B Scale Verification (Task 13)

The ultimate goal is to compare Llama3 with and without AttnRes at the same
compute budget and measure the loss improvement.

### Config Alignment (audited)

The AttnRes model IS Llama3 architecture with residual connections replaced by
block attention residuals. For a fair comparison, all architecture and training
parameters must be identical.

| Parameter | Llama3 `debugmodel` | AttnRes `debugmodel` | Match? |
|-----------|---------------------|----------------------|--------|
| dim | 256 | 256 | Yes |
| n_layers | 6 | 6 | Yes |
| vocab_size | 2048 | 2048 | Yes |
| n_heads | 16 | 16 | Yes |
| ffn_hidden_dim | 256 | 256 | Yes |
| lr | 8e-4 | 8e-4 | Yes |
| batch_size | 8 | 8 | Yes |
| seq_len | 2048 | 2048 | Yes |
| dataset | c4_test | c4_test | Yes |

**1B gap**: Fixed. Added `enable_weight_tying=True` to AttnRes 1B config and
implemented full weight tying support in `AttnResDecoder`. 7 tests verify
correctness (shared tensor, param count reduction, init_weights survival,
forward/backward, config flags, ModelSpec).

### Comparison Commands

```bash
# Step 1: Verify parallelism produces identical loss (prerequisite)
# Run 1-GPU, 2-GPU FSDP, 4-GPU FSDP+TP and compare losses
torchrun --nproc_per_node=1 -m torchtitan.train \
    --module attn_residuals --config attn_res_debugmodel \
    --training.steps 20 --debug.seed 42 --debug.deterministic \
    --metrics.log_dir ./logs/attnres_1gpu

# Step 2: Run Llama3 baseline
torchrun --nproc_per_node=NUM_GPUS -m torchtitan.train \
    --module llama3 --config llama3_debugmodel \
    --training.steps 500 --debug.seed 42 --debug.deterministic \
    --metrics.log_dir ./logs/llama3_baseline

# Step 3: Run AttnRes
torchrun --nproc_per_node=NUM_GPUS -m torchtitan.train \
    --module attn_residuals --config attn_res_debugmodel \
    --training.steps 500 --debug.seed 42 --debug.deterministic \
    --metrics.log_dir ./logs/attnres

# Step 4: Compare via TensorBoard
tensorboard --logdir ./logs
```

### 1B Comparison (8 GPUs)

```bash
# Using c4_test (local, always works)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py \
    --task 12.4a --steps 1000

# Using full C4 (requires HF network access)
python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py \
    --task 12.4b --steps 1000

# Or run manually:
torchrun --nproc_per_node=8 -m torchtitan.train \
    --module attn_residuals --config llama3_1b_baseline \
    --training.steps 1000 --debug.seed 42 --debug.deterministic \
    --parallelism.data_parallel_shard_degree 8 \
    --metrics.enable_tensorboard --metrics.save_tb_folder tb_llama3_1b \
    --dump_folder ./outputs/attnres_1b_compare

torchrun --nproc_per_node=8 -m torchtitan.train \
    --module attn_residuals --config attn_res_1b \
    --training.steps 1000 --debug.seed 42 --debug.deterministic \
    --parallelism.data_parallel_shard_degree 8 \
    --metrics.enable_tensorboard --metrics.save_tb_folder tb_attnres_1b \
    --dump_folder ./outputs/attnres_1b_compare

# Compare via TensorBoard
tensorboard --logdir ./outputs/attnres_1b_compare
```

### Loss Comparison Results

![Loss Comparison](loss_comparison_combined.png)

**debugmodel** (c4_test, 500 steps, 1 GPU): Llama3 wins — too few layers/blocks
for depth-selective attention.

**1B on c4_test** (1000 steps, 8 GPUs FSDP): AttnRes 5.1% lower avg loss
(last 50 steps). Both models memorize the tiny dataset.

**1B on full C4** (1000 steps, 8 GPUs FSDP): **AttnRes consistently lower from
step ~100** (99-100% of steps). 1.0% lower avg loss — cleaner generalization
signal. TPS overhead 36%, memory overhead 0.2%.

See [REPORT.md](REPORT.md) for detailed tables and analysis.

### 8B Comparison (8 GPUs, full C4)

The paper's main claims (1.25x compute equivalence, <4% TPS overhead) are at 7B+
scale. The 8B comparison verifies these at Llama3 8B scale (dim=4096, 32 layers).

**Important**: For long 8B runs, increase the NCCL timeout from the default 100s
to 300s (`--comm.train_timeout_seconds 300`) and the HuggingFace download timeout
(`HF_HUB_DOWNLOAD_TIMEOUT=120`). This prevents transient network stalls from
killing multi-hour training runs.

```bash
# Automated (requires HF network access to cas-bridge.xethub.hf.co)
HF_HUB_DOWNLOAD_TIMEOUT=120 python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py \
    --task 13 --steps 5000 --output-dir ./outputs/attnres_8b_compare

# Or run manually:
HF_HUB_DOWNLOAD_TIMEOUT=120 torchrun --nproc_per_node=8 -m torchtitan.train \
    --module attn_residuals --config llama3_8b_baseline \
    --training.steps 5000 --debug.seed 42 --debug.deterministic \
    --parallelism.data_parallel_shard_degree 8 \
    --comm.train_timeout_seconds 300 \
    --metrics.enable_tensorboard --metrics.save_tb_folder tb_llama3_8b \
    --dump_folder ./outputs/attnres_8b_compare --metrics.log_freq 1

HF_HUB_DOWNLOAD_TIMEOUT=120 torchrun --nproc_per_node=8 -m torchtitan.train \
    --module attn_residuals --config attn_res_8b \
    --training.steps 5000 --debug.seed 42 --debug.deterministic \
    --parallelism.data_parallel_shard_degree 8 \
    --comm.train_timeout_seconds 300 \
    --metrics.enable_tensorboard --metrics.save_tb_folder tb_attnres_8b \
    --dump_folder ./outputs/attnres_8b_compare --metrics.log_freq 1
```

**Resuming from checkpoint**: If a run crashes mid-training, resume with
`--checkpoint.initial_load_path` and **`--checkpoint.no_initial_load_model_only`**.
The second flag is critical — without it, only model weights are loaded and the
optimizer, LR scheduler, dataloader position, and step counter are all reset.

```bash
# Example: resume Llama3 8B from step 3000 checkpoint
HF_HUB_DOWNLOAD_TIMEOUT=120 torchrun --nproc_per_node=8 -m torchtitan.train \
    --module attn_residuals --config llama3_8b_baseline \
    --training.steps 5000 --debug.seed 42 --debug.deterministic \
    --parallelism.data_parallel_shard_degree 8 \
    --comm.train_timeout_seconds 300 \
    --metrics.enable_tensorboard --metrics.save_tb_folder tb_llama3_8b \
    --dump_folder ./outputs/attnres_8b_compare --metrics.log_freq 1 \
    --checkpoint.initial_load_path ./outputs/attnres_8b_compare/checkpoints/step-3000 \
    --checkpoint.no_initial_load_model_only
```

| Parameter | Llama3 8B | AttnRes 8B |
|-----------|-----------|------------|
| dim | 4096 | 4096 |
| n_layers | 32 | 32 |
| num_blocks | -- | 16 (2 layers/block) |
| n_heads / n_kv_heads | 32 / 8 | 32 / 8 |
| ffn_hidden_dim | 14,336 | 14,336 |
| lr | 3e-4 | 3e-4 |
| batch (local/global) | 1 / 8 | 1 / 8 |
| seq_len | 8192 | 8192 |
| steps | 5000 | 5000 |
| dataset | c4 (full) | c4 (full) |
| AC | selective | selective |

**Expected results** (from paper at 7B+ scale):
- Steps-to-target-loss ratio: ~1.25x (Llama3 needs 25% more steps)
- TPS overhead: <4%
- Memory overhead: <1%

### What to Expect and Measure

TorchTitan already logs all necessary metrics to TensorBoard (`--metrics.enable_tensorboard`).
The comparison has three dimensions:

**Primary: Convergence efficiency (the paper's main claim)**
- `loss_metrics/global_avg_loss`: AttnRes should reach lower loss at same step count
- Compute-to-target-loss: baseline should need **~1.25x more steps** to reach the
  same loss that AttnRes achieves
- `grad_norm`: AttnRes should show more uniform values across layers

**Secondary: Per-step overhead (should be negligible)**
- `throughput(tps)`: tokens/sec should be within ~4% of baseline (paper §4.1)
- `tflops` / `mfu(%)`: MFU should be nearly identical

**Tertiary: Memory overhead (small increase expected)**
- `memory/max_active(GiB)`: AttnRes stores N block representations [B,T,D],
  so peak memory will be slightly higher than baseline

**Important clarification**: The paper does NOT claim AttnRes uses less compute
or memory per step. AttnRes uses *slightly more* per step (block storage +
softmax attention over depth). The claim is that AttnRes **converges faster** —
it reaches the same loss quality in fewer training steps/tokens.

- At debugmodel scale (dim=256), gains may be small but directionally correct
- At 1B scale, gains should be more pronounced

### Remaining Work (Task 13: 8B Verification)

| Step | Description | Status |
|------|-------------|--------|
| 13.1 | Create AttnRes 8B model config (match Llama3 8B) | ✅ Complete |
| 13.2 | Create 8B trainer configs (attn_res_8b, llama3_8b_baseline) | ✅ Complete |
| 13.3 | Add 8B task to verify_parallelism.py | ✅ Complete |
| 13.4 | Run Llama3 8B baseline (5000+ steps, full C4) | Pending |
| 13.5 | Run AttnRes 8B (5000+ steps, full C4) | Pending |
| 13.6 | Compute steps-to-target-loss ratio (expect ~1.25x) | Pending |
| 13.7 | Compare TPS overhead (expect <4%) | Pending |
| 13.8 | Generate loss plots and update report | Pending |
