#!/bin/bash
# Scaled runs at a given step count. Usage: bash scaled_runs.sh <STEPS>
# 8 unique runs feed all plots (base config covers tier0 + offpolicy3 + compile_on).
set -uo pipefail
source rl_eval/activate_env.sh
export no_proxy="localhost,127.0.0.1,.internalfb.com,.facebook.com,.fbcdn.net,.tfbnw.net,.fb.com,.fbinfra.net"; export NO_PROXY="$no_proxy"
S=${1:?need step count}
T="python -m torchtitan.experiments.rl.train --metrics.no-enable-wandb --async-loop.num-training-steps $S"
run(){ local tag=$1; shift; echo "===== $tag START $(date -u +%T) (steps=$S) ====="; rm -rf outputs/rl/checkpoint 2>/dev/null; "$@" > rl_eval/logs/scaled_${tag}_s${S}.log 2>&1; echo "===== $tag DONE $(date -u +%T) rc=$? ====="; }

run base_ops3_compon  $T --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen
run ops0              $T --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen --async-loop.max-offpolicy-steps 0
run ops1              $T --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen --async-loop.max-offpolicy-steps 1
run compoff           $T --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen --compile.no-enable
run bi_on             $T --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen_batch_invariant
run bi_off            $T --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen_batch_invariant --trainer.debug.no-batch-invariant --generator.debug.no-batch-invariant --trainer.debug.no-deterministic
run cl_grpo           $T --module count_letters --config rl_grpo_qwen3_0_6b_count_letters
run cl_dapo           $T --module count_letters --config rl_grpo_qwen3_0_6b_count_letters_dapo
echo "ALL ${S}-STEP RUNS DONE $(date -u)"
