#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelism and comparison verification for AttnRes models.

Runs training with different parallelism configs and verifies:
1. Determinism: repeated runs with the same seed produce bitwise identical loss
2. Convergence: loss decreases over training steps
3. No NaN/Inf in loss values
4. Model comparison: Llama3 vs AttnRes loss, throughput, and memory

Usage (from repo root, with conda env 'titan' activated):

    # Run all determinism verification tasks (12.2a-c)
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py

    # Run a specific task
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.2a

    # Run Llama3 vs AttnRes debugmodel comparison (500 steps by default)
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.3a --steps 500

    # Run Llama3 vs AttnRes 1B on c4_test (1000 steps by default)
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.4a --steps 1000

    # Run Llama3 vs AttnRes 1B on full C4 (1000 steps by default)
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 12.4b --steps 1000

    # Run Llama3 vs AttnRes 8B on full C4 (5000 steps by default)
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --task 13 --steps 5000

    # Custom output directory
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --output-dir ./my_outputs

    # Custom number of steps
    python torchtitan/experiments/attn_residuals/tests/verify_parallelism.py --steps 50

Prerequisites:
    - conda environment 'titan' with torchtitan installed
    - At least 1 GPU for 12.3a, 2 GPUs for FSDP/TP, 4 GPUs for FSDP+TP
    - tensorboard package installed (pip install tensorboard)
"""

import argparse
import math
import os
import shutil
import statistics
import subprocess
import sys


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMMON_ARGS = [
    "--module",
    "attn_residuals",
    "--config",
    "attn_res_debugmodel",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

LLAMA3_COMMON_ARGS = [
    "--module",
    "llama3",
    "--config",
    "llama3_debugmodel",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

ATTNRES_COMMON_ARGS = COMMON_ARGS  # alias for clarity

# 1B configs (c4_test): both use --module attn_residuals (Llama3 baseline
# config is registered in the AttnRes config_registry to avoid modifying core).
LLAMA3_1B_COMMON_ARGS = [
    "--module",
    "attn_residuals",
    "--config",
    "llama3_1b_baseline",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

ATTNRES_1B_COMMON_ARGS = [
    "--module",
    "attn_residuals",
    "--config",
    "attn_res_1b",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

# 1B configs (full C4): identical to above but using streamed C4 dataset.
LLAMA3_1B_C4_COMMON_ARGS = [
    "--module",
    "attn_residuals",
    "--config",
    "llama3_1b_baseline_c4",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

ATTNRES_1B_C4_COMMON_ARGS = [
    "--module",
    "attn_residuals",
    "--config",
    "attn_res_1b_c4",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

# 8B configs (full C4): for verifying the paper's 1.25x compute claim at scale.
LLAMA3_8B_COMMON_ARGS = [
    "--module",
    "attn_residuals",
    "--config",
    "llama3_8b_baseline",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

ATTNRES_8B_COMMON_ARGS = [
    "--module",
    "attn_residuals",
    "--config",
    "attn_res_8b",
    "--debug.seed",
    "42",
    "--debug.deterministic",
    "--metrics.enable_tensorboard",
    "--validator.freq",
    "0",
]

# Tasks 12.2a-c: determinism verification
# Task 12.3a: Llama3 vs AttnRes debugmodel comparison
# Task 12.4a: Llama3 vs AttnRes 1B comparison (c4_test)
# Task 12.4b: Llama3 vs AttnRes 1B comparison (full C4)
COMPARISON_TASKS = {
    "12.3a": {
        "name": "Llama3 vs AttnRes debugmodel Comparison",
        "default_steps": 500,
        "runs": [
            {
                "label": "llama3_baseline",
                "ngpu": 1,
                "common_args": LLAMA3_COMMON_ARGS,
                "extra_args": ["--metrics.log_freq", "1"],
                "local_batch_size": 8,
            },
            {
                "label": "attnres",
                "ngpu": 1,
                "common_args": ATTNRES_COMMON_ARGS,
                "extra_args": ["--metrics.log_freq", "1"],
                "local_batch_size": 8,
            },
        ],
        "compare_pairs": [("llama3_baseline", "attnres")],
    },
    "12.4a": {
        "name": "Llama3 vs AttnRes 1B Comparison",
        "default_steps": 1000,
        "runs": [
            {
                "label": "llama3_1b",
                "ngpu": 8,
                "common_args": LLAMA3_1B_COMMON_ARGS,
                "extra_args": [
                    "--metrics.log_freq",
                    "1",
                    "--parallelism.data_parallel_shard_degree",
                    "8",
                ],
                "local_batch_size": 2,
            },
            {
                "label": "attnres_1b",
                "ngpu": 8,
                "common_args": ATTNRES_1B_COMMON_ARGS,
                "extra_args": [
                    "--metrics.log_freq",
                    "1",
                    "--parallelism.data_parallel_shard_degree",
                    "8",
                ],
                "local_batch_size": 2,
            },
        ],
        "compare_pairs": [("llama3_1b", "attnres_1b")],
    },
    "12.4b": {
        "name": "Llama3 vs AttnRes 1B Comparison (full C4)",
        "default_steps": 1000,
        "runs": [
            {
                "label": "llama3_1b_c4",
                "ngpu": 8,
                "common_args": LLAMA3_1B_C4_COMMON_ARGS,
                "extra_args": [
                    "--metrics.log_freq",
                    "1",
                    "--parallelism.data_parallel_shard_degree",
                    "8",
                ],
                "local_batch_size": 2,
            },
            {
                "label": "attnres_1b_c4",
                "ngpu": 8,
                "common_args": ATTNRES_1B_C4_COMMON_ARGS,
                "extra_args": [
                    "--metrics.log_freq",
                    "1",
                    "--parallelism.data_parallel_shard_degree",
                    "8",
                ],
                "local_batch_size": 2,
            },
        ],
        "compare_pairs": [("llama3_1b_c4", "attnres_1b_c4")],
    },
    "13": {
        "name": "Llama3 vs AttnRes 8B Comparison (full C4)",
        "default_steps": 5000,
        "runs": [
            {
                "label": "llama3_8b",
                "ngpu": 8,
                "common_args": LLAMA3_8B_COMMON_ARGS,
                "extra_args": [
                    "--metrics.log_freq",
                    "1",
                    "--parallelism.data_parallel_shard_degree",
                    "8",
                    "--comm.train_timeout_seconds",
                    "300",
                ],
                "local_batch_size": 1,
            },
            {
                "label": "attnres_8b",
                "ngpu": 8,
                "common_args": ATTNRES_8B_COMMON_ARGS,
                "extra_args": [
                    "--metrics.log_freq",
                    "1",
                    "--parallelism.data_parallel_shard_degree",
                    "8",
                    "--comm.train_timeout_seconds",
                    "300",
                ],
                "local_batch_size": 1,
            },
        ],
        "compare_pairs": [("llama3_8b", "attnres_8b")],
    },
}

TASKS = {
    "12.2a": {
        "name": "FSDP Determinism",
        "runs": [
            {
                "label": "1gpu",
                "ngpu": 1,
                "extra_args": [],
                "local_batch_size": 8,
            },
            {
                "label": "fsdp_run1",
                "ngpu": 2,
                "extra_args": [
                    "--parallelism.data_parallel_shard_degree",
                    "2",
                ],
                "local_batch_size": 4,
            },
            {
                "label": "fsdp_run2",
                "ngpu": 2,
                "extra_args": [
                    "--parallelism.data_parallel_shard_degree",
                    "2",
                ],
                "local_batch_size": 4,
            },
        ],
        "determinism_pairs": [("fsdp_run1", "fsdp_run2")],
        "convergence_runs": ["1gpu", "fsdp_run1"],
    },
    "12.2b": {
        "name": "TP Determinism",
        "runs": [
            {
                "label": "tp_run1",
                "ngpu": 2,
                "extra_args": [
                    "--parallelism.tensor_parallel_degree",
                    "2",
                ],
                "local_batch_size": 8,
            },
            {
                "label": "tp_run2",
                "ngpu": 2,
                "extra_args": [
                    "--parallelism.tensor_parallel_degree",
                    "2",
                ],
                "local_batch_size": 8,
            },
        ],
        "determinism_pairs": [("tp_run1", "tp_run2")],
        "convergence_runs": ["tp_run1"],
    },
    "12.2c": {
        "name": "FSDP+TP Determinism",
        "runs": [
            {
                "label": "fsdp_tp_run1",
                "ngpu": 4,
                "extra_args": [
                    "--parallelism.data_parallel_shard_degree",
                    "2",
                    "--parallelism.tensor_parallel_degree",
                    "2",
                ],
                "local_batch_size": 4,
            },
            {
                "label": "fsdp_tp_run2",
                "ngpu": 4,
                "extra_args": [
                    "--parallelism.data_parallel_shard_degree",
                    "2",
                    "--parallelism.tensor_parallel_degree",
                    "2",
                ],
                "local_batch_size": 4,
            },
        ],
        "determinism_pairs": [("fsdp_tp_run1", "fsdp_tp_run2")],
        "convergence_runs": ["fsdp_tp_run1"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_training(
    label, ngpu, extra_args, local_batch_size, steps, output_dir, common_args=None
):
    """Run a single training job via torchrun and return the TB directory."""
    if common_args is None:
        common_args = COMMON_ARGS
    tb_folder = f"tb_{label}"
    cmd = [
        "torchrun",
        f"--nproc_per_node={ngpu}",
        "-m",
        "torchtitan.train",
        *common_args,
        "--training.steps",
        str(steps),
        "--training.local_batch_size",
        str(local_batch_size),
        "--metrics.save_tb_folder",
        tb_folder,
        "--dump_folder",
        output_dir,
        *extra_args,
    ]

    # Clear stale TB data
    tb_dir = os.path.join(output_dir, tb_folder)
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)

    print(f"\n{'=' * 70}")
    print(f"Running: {label}  (ngpu={ngpu}, local_batch={local_batch_size})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {label}")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)

    # Print loss lines from stdout
    for line in result.stderr.split("\n"):
        if "step:" in line and "loss:" in line:
            # Only print from rank 0 (first occurrence per step)
            print(line)

    return tb_dir


def extract_tb_scalars(tb_base_dir, tag):
    """Extract scalar values from TensorBoard event files."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    subdirs = [
        d
        for d in os.listdir(tb_base_dir)
        if os.path.isdir(os.path.join(tb_base_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories in {tb_base_dir}")
    event_dir = os.path.join(tb_base_dir, subdirs[-1])

    ea = EventAccumulator(event_dir)
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    if tag not in available:
        raise KeyError(f"Tag '{tag}' not found. Available: {available}")

    return {s.step: s.value for s in ea.Scalars(tag)}


def check_determinism(losses_a, losses_b, grads_a, grads_b, label_a, label_b):
    """Check if two runs produced bitwise identical loss and grad_norm."""
    print(f"\nDeterminism check: {label_a} vs {label_b}")
    print(f"{'Step':>4}  {'Loss A':>20}  {'Loss B':>20}  Match")
    print(f"{'----':>4}  {'------':>20}  {'------':>20}  -----")

    all_match = True
    for step in sorted(losses_a.keys()):
        la = losses_a[step]
        lb = losses_b.get(step, float("nan"))
        match = la == lb
        if not match:
            all_match = False
        print(
            f"{step:4d}  {repr(la):>20s}  {repr(lb):>20s}  {'YES' if match else 'NO'}"
        )

    grad_match = all(grads_a.get(s) == grads_b.get(s) for s in grads_a)

    print(f"\nLoss bitwise identical:      {all_match}")
    print(f"Grad_norm bitwise identical: {grad_match}")
    return all_match and grad_match


def check_convergence(losses, label):
    """Check that loss decreases and contains no NaN/Inf."""
    steps = sorted(losses.keys())
    first_loss = losses[steps[0]]
    last_loss = losses[steps[-1]]
    converges = last_loss < first_loss
    has_nan = any(math.isnan(v) or math.isinf(v) for v in losses.values())
    reduction_pct = (1 - last_loss / first_loss) * 100

    print(f"\nConvergence check: {label}")
    print(f"  Loss step {steps[0]}: {first_loss:.6f}")
    print(f"  Loss step {steps[-1]}: {last_loss:.6f}")
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Converges: {converges}")
    print(f"  No NaN/Inf: {not has_nan}")
    return converges and not has_nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_task(task_id, steps, output_dir):
    """Run a single verification task."""
    task = TASKS[task_id]
    print(f"\n{'#' * 70}")
    print(f"# Task {task_id}: {task['name']}")
    print(f"# Steps: {steps}, Output: {output_dir}")
    print(f"{'#' * 70}")

    # Run all training jobs
    tb_dirs = {}
    for run in task["runs"]:
        tb_dir = run_training(
            label=run["label"],
            ngpu=run["ngpu"],
            extra_args=run["extra_args"],
            local_batch_size=run["local_batch_size"],
            steps=steps,
            output_dir=output_dir,
        )
        tb_dirs[run["label"]] = tb_dir

    # Extract losses and grad_norms
    all_losses = {}
    all_grads = {}
    for label, tb_dir in tb_dirs.items():
        all_losses[label] = extract_tb_scalars(tb_dir, "loss_metrics/global_avg_loss")
        try:
            all_grads[label] = extract_tb_scalars(tb_dir, "grad_norm")
        except KeyError:
            all_grads[label] = {}

    # Determinism checks
    det_pass = True
    for label_a, label_b in task["determinism_pairs"]:
        ok = check_determinism(
            all_losses[label_a],
            all_losses[label_b],
            all_grads[label_a],
            all_grads[label_b],
            label_a,
            label_b,
        )
        if not ok:
            det_pass = False

    # Convergence checks
    conv_pass = True
    for label in task["convergence_runs"]:
        ok = check_convergence(all_losses[label], label)
        if not ok:
            conv_pass = False

    # Summary
    passed = det_pass and conv_pass
    print(f"\n{'=' * 70}")
    print(f"Task {task_id} ({task['name']}): {'PASS' if passed else 'FAIL'}")
    print(f"  Determinism: {'PASS' if det_pass else 'FAIL'}")
    print(f"  Convergence: {'PASS' if conv_pass else 'FAIL'}")
    print(f"{'=' * 70}")
    return passed


def check_comparison(all_metrics, label_a, label_b):
    """Compare two models across loss, throughput, and memory."""
    losses_a = all_metrics[label_a]["loss_metrics/global_avg_loss"]
    losses_b = all_metrics[label_b]["loss_metrics/global_avg_loss"]

    # --- Loss comparison at milestones ---
    print(f"\n{'=' * 80}")
    print(f"LOSS COMPARISON: {label_a} vs {label_b}")
    print(f"{'=' * 80}")
    print(
        f"{'Step':>5} {label_a + ' Loss':>20} {label_b + ' Loss':>20} "
        f"{'Diff':>12} {'Better':>8}"
    )

    steps_a = sorted(losses_a.keys())
    milestones = [
        s
        for s in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
        if s <= steps_a[-1]
    ]
    for step in milestones:
        la = losses_a.get(step)
        lb = losses_b.get(step)
        if la is not None and lb is not None:
            diff = lb - la
            better = label_a if la < lb else label_b if lb < la else "TIE"
            print(f"{step:5d} {la:20.6f} {lb:20.6f} {diff:12.6f} {better:>8}")

    # --- Average loss over last 50 steps ---
    last_n = 50
    cutoff = steps_a[-1] - last_n
    avg_a = statistics.mean([v for s, v in losses_a.items() if s > cutoff])
    avg_b = statistics.mean([v for s, v in losses_b.items() if s > cutoff])
    print(f"\nAvg loss (last {last_n} steps):")
    print(f"  {label_a}: {avg_a:.6f}")
    print(f"  {label_b}: {avg_b:.6f}")
    print(
        f"  Difference: {avg_b - avg_a:.6f} "
        f"({'better' if avg_b < avg_a else 'worse'} for {label_b})"
    )

    # --- Throughput comparison ---
    tps_a = all_metrics[label_a].get("throughput(tps)", {})
    tps_b = all_metrics[label_b].get("throughput(tps)", {})
    if tps_a and tps_b:
        # Skip warmup (first 10 steps)
        avg_tps_a = statistics.mean([v for s, v in tps_a.items() if s >= 10])
        avg_tps_b = statistics.mean([v for s, v in tps_b.items() if s >= 10])
        overhead = (1 - avg_tps_b / avg_tps_a) * 100

        print(f"\n{'=' * 80}")
        print("THROUGHPUT COMPARISON")
        print(f"{'=' * 80}")
        print(f"  {label_a} avg TPS: {avg_tps_a:,.0f}")
        print(f"  {label_b} avg TPS: {avg_tps_b:,.0f}")
        print(f"  Overhead: {overhead:.1f}%")

    # --- Memory comparison ---
    mem_a = all_metrics[label_a].get("memory/max_active(GiB)", {})
    mem_b = all_metrics[label_b].get("memory/max_active(GiB)", {})
    if mem_a and mem_b:
        peak_a = max(mem_a.values())
        peak_b = max(mem_b.values())
        mem_overhead = (peak_b / peak_a - 1) * 100

        print(f"\n{'=' * 80}")
        print("MEMORY COMPARISON")
        print(f"{'=' * 80}")
        print(f"  {label_a} peak active: {peak_a:.4f} GiB")
        print(f"  {label_b} peak active: {peak_b:.4f} GiB")
        print(f"  Overhead: {mem_overhead:.1f}%")

    # --- Time per step ---
    time_a = all_metrics[label_a].get("time_metrics/end_to_end(s)", {})
    time_b = all_metrics[label_b].get("time_metrics/end_to_end(s)", {})
    if time_a and time_b:
        avg_time_a = statistics.mean([v for s, v in time_a.items() if s >= 10])
        avg_time_b = statistics.mean([v for s, v in time_b.items() if s >= 10])
        time_overhead = (avg_time_b / avg_time_a - 1) * 100

        print(f"\n{'=' * 80}")
        print("TIME PER STEP")
        print(f"{'=' * 80}")
        print(f"  {label_a} avg time/step: {avg_time_a * 1000:.2f} ms")
        print(f"  {label_b} avg time/step: {avg_time_b * 1000:.2f} ms")
        print(f"  Overhead: {time_overhead:.1f}%")

    return True  # Comparison tasks always "pass" — they report, not assert


def plot_losses(all_metrics, label_a, label_b, output_path, title=None):
    """Plot loss curves for two models and save to a PNG file."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping loss plot.")
        return None

    losses_a = all_metrics[label_a]["loss_metrics/global_avg_loss"]
    losses_b = all_metrics[label_b]["loss_metrics/global_avg_loss"]

    steps_a = sorted(losses_a.keys())
    steps_b = sorted(losses_b.keys())
    vals_a = [losses_a[s] for s in steps_a]
    vals_b = [losses_b[s] for s in steps_b]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps_a, vals_a, label=label_a, linewidth=1.5)
    ax.plot(steps_b, vals_b, label=label_b, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title or f"Loss: {label_a} vs {label_b}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nLoss plot saved to: {output_path}")
    return output_path


def run_comparison_task(task_id, steps, output_dir):
    """Run a model comparison task (e.g., Llama3 vs AttnRes)."""
    task = COMPARISON_TASKS[task_id]
    if steps is None:
        steps = task.get("default_steps", 500)

    print(f"\n{'#' * 70}")
    print(f"# Task {task_id}: {task['name']}")
    print(f"# Steps: {steps}, Output: {output_dir}")
    print(f"{'#' * 70}")

    # Run all training jobs
    tb_dirs = {}
    for run in task["runs"]:
        tb_dir = run_training(
            label=run["label"],
            ngpu=run["ngpu"],
            extra_args=run["extra_args"],
            local_batch_size=run["local_batch_size"],
            steps=steps,
            output_dir=output_dir,
            common_args=run.get("common_args"),
        )
        tb_dirs[run["label"]] = tb_dir

    # Extract all metrics
    tags = [
        "loss_metrics/global_avg_loss",
        "throughput(tps)",
        "memory/max_active(GiB)",
        "time_metrics/end_to_end(s)",
        "mfu(%)",
        "grad_norm",
    ]
    all_metrics = {}
    for label, tb_dir in tb_dirs.items():
        all_metrics[label] = {}
        for tag in tags:
            try:
                all_metrics[label][tag] = extract_tb_scalars(tb_dir, tag)
            except (KeyError, FileNotFoundError):
                all_metrics[label][tag] = {}

    # Run comparisons and generate plots
    for label_a, label_b in task["compare_pairs"]:
        check_comparison(all_metrics, label_a, label_b)
        plot_path = os.path.join(output_dir, f"loss_{label_a}_vs_{label_b}.png")
        plot_losses(
            all_metrics,
            label_a,
            label_b,
            plot_path,
            title=f"{task['name']} — Loss Curves",
        )

    # Convergence check for both
    conv_pass = True
    for run in task["runs"]:
        label = run["label"]
        losses = all_metrics[label].get("loss_metrics/global_avg_loss", {})
        if losses:
            ok = check_convergence(losses, label)
            if not ok:
                conv_pass = False

    print(f"\n{'=' * 70}")
    print(f"Task {task_id} ({task['name']}): COMPLETE")
    print(f"  Convergence: {'PASS' if conv_pass else 'FAIL'}")
    print(f"{'=' * 70}")
    return conv_pass


def main():
    all_task_ids = list(TASKS.keys()) + list(COMPARISON_TASKS.keys())
    parser = argparse.ArgumentParser(
        description="Parallelism and comparison verification for AttnRes",
    )
    parser.add_argument(
        "--task",
        choices=all_task_ids + ["all", "all_determinism"],
        default="all_determinism",
        help="Which task to run. 'all_determinism' runs 12.2a-c, "
        "'all' runs everything including comparisons (default: all_determinism)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of training steps per run "
        "(default: 20 for determinism, 500 for comparison)",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/attnres_verify",
        help="Output directory for TensorBoard logs (default: ./outputs/attnres_verify)",
    )
    args = parser.parse_args()

    if args.task == "all":
        tasks_to_run = all_task_ids
    elif args.task == "all_determinism":
        tasks_to_run = list(TASKS.keys())
    else:
        tasks_to_run = [args.task]

    results = {}
    for task_id in tasks_to_run:
        if task_id in TASKS:
            steps = args.steps if args.steps is not None else 20
            results[task_id] = run_task(task_id, steps, args.output_dir)
        elif task_id in COMPARISON_TASKS:
            results[task_id] = run_comparison_task(task_id, args.steps, args.output_dir)

    # Final summary
    print(f"\n{'#' * 70}")
    print("# FINAL SUMMARY")
    print(f"{'#' * 70}")
    all_passed = True
    all_tasks_dict = {**TASKS, **COMPARISON_TASKS}
    for task_id, passed in results.items():
        status = "PASS" if passed else "FAIL"
        name = all_tasks_dict[task_id]["name"]
        print(f"  Task {task_id} ({name}): {status}")
        if not passed:
            all_passed = False
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
