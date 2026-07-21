#!/bin/bash
# Option A: 1000-step learning curves for Tier 0 (varlen) + Tier 2 (count_letters).
# Waits for the 100-step batch to finish first so GPUs don't collide.
set -uo pipefail
cd /home/alisol/projects/torchtitan

# 1. Wait for the 100-step batch to complete (its pid file)
if [ -f rl_eval/logs/scaled100.pid ]; then
  P=$(cat rl_eval/logs/scaled100.pid)
  echo "[wait] 100-step batch pid=$P — waiting for it to finish $(date -u)"
  while ps -p "$P" >/dev/null 2>&1; do sleep 60; done
  echo "[wait] 100-step batch done $(date -u)"
fi
# extra safety: wait until no titan train process is running
while pgrep -f "experiments.rl.train" >/dev/null 2>&1; do sleep 30; done
echo "[wait] GPUs clear, starting 1000-step runs $(date -u)"

source rl_eval/activate_env.sh
export no_proxy="localhost,127.0.0.1,.internalfb.com,.facebook.com,.fbcdn.net,.tfbnw.net,.fb.com,.fbinfra.net"; export NO_PROXY="$no_proxy"
S=1000
run(){ local tag=$1; shift; echo "===== $tag START $(date -u +%T) (steps=$S) ====="; rm -rf outputs/rl/checkpoint 2>/dev/null; "$@" > rl_eval/logs/scaled_${tag}_s${S}.log 2>&1; echo "===== $tag DONE $(date -u +%T) rc=$? ====="; }
T="python -m torchtitan.experiments.rl.train --metrics.no-enable-wandb --async-loop.num-training-steps $S"

run base_ops3_compon $T --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen           # Tier 0 curve @ 1000
run cl_grpo          $T --module count_letters --config rl_grpo_qwen3_0_6b_count_letters    # Tier 2 curve @ 1000
echo "ALL 1000-STEP RUNS DONE $(date -u)"
