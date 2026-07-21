#!/bin/bash
# Re-run the 2 count_letters runs that the G9 hang killed: cl_grpo + cl_dapo @ 100 steps.
# Waits for the 1000-step overnight job (all-GPU) to finish first. Uses the watchdog (stall=900s).
set -uo pipefail
cd /home/alisol/projects/torchtitan
# wait for 1000-step job
if [ -f rl_eval/logs/scaled1000.pid ]; then
  P=$(cat rl_eval/logs/scaled1000.pid)
  echo "[wait] 1000-step job pid=$P — waiting $(date -u)"
  while ps -p "$P" >/dev/null 2>&1; do sleep 60; done
fi
while pgrep -f "experiments.rl.train" >/dev/null 2>&1; do sleep 30; done
echo "[wait] GPUs clear, re-running count_letters $(date -u)"
source rl_eval/activate_env.sh
export no_proxy="localhost,127.0.0.1,.internalfb.com,.facebook.com,.fbcdn.net,.tfbnw.net,.fb.com,.fbinfra.net"; export NO_PROXY="$no_proxy"
WD=rl_eval/run_with_watchdog.sh
rm -rf outputs/rl/checkpoint
bash $WD rl_eval/logs/scaled_cl_grpo_s100.log 900 \
  python -m torchtitan.experiments.rl.train --module count_letters \
  --config rl_grpo_qwen3_0_6b_count_letters --metrics.no-enable-wandb --async-loop.num-training-steps 100
echo "===== cl_grpo rerun rc=$? $(date -u +%T) ====="
rm -rf outputs/rl/checkpoint
bash $WD rl_eval/logs/scaled_cl_dapo_s100.log 900 \
  python -m torchtitan.experiments.rl.train --module count_letters \
  --config rl_grpo_qwen3_0_6b_count_letters_dapo --metrics.no-enable-wandb --async-loop.num-training-steps 100
echo "===== cl_dapo rerun rc=$? $(date -u +%T) ====="
echo "COUNT_LETTERS RERUN DONE $(date -u)"
