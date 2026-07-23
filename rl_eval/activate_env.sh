#!/bin/bash
# Source this to activate the TitanRL eval env with the cuBLAS ABI fix.
#   source rl_eval/activate_env.sh
ROOT="/home/alisol/projects/torchtitan"
source "$ROOT/venv_titanrl/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
# vLLM nightly stable-ABI .so needs cuBLAS symbols globally visible (torch loads them RTLD_LOCAL).
# Preload the same nvidia-pip cuBLAS torch uses. Inherited by Monarch-spawned workers.
_NVLIB="$ROOT/venv_titanrl/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_PRELOAD="$_NVLIB/libcublasLt.so.13:$_NVLIB/libcublas.so.13:${LD_PRELOAD:-}"
export HF_ASSETS_PATH_DEFAULT="$ROOT/torchtitan/experiments/rl/example_checkpoint"

# W&B logging (user preference: always log to their W&B). Logged in as a-shamsoshoara.
export WANDB_PROJECT="${WANDB_PROJECT:-titanrl-eval}"
# runs use --metrics.enable-wandb (default ON); do NOT pass --metrics.no-enable-wandb
