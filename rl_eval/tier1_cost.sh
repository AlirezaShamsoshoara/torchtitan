#!/bin/bash
# Clean batch-invariant ON vs OFF cost comparison, same config/8-GPU/steps, fresh checkpoint each.
source rl_eval/activate_env.sh
export no_proxy="localhost,127.0.0.1,.internalfb.com,.facebook.com,.fbcdn.net,.tfbnw.net,.fb.com,.fbinfra.net"; export NO_PROXY="$no_proxy"
CFG=rl_grpo_qwen3_0_6b_varlen_batch_invariant
STEPS=6
run() {
  local tag="$1"; shift
  echo "===== $tag START $(date -u +%H:%M:%S) ====="
  rm -rf outputs/rl/checkpoint outputs/rl/structured_logs 2>/dev/null
  /usr/bin/time -v python -m torchtitan.experiments.rl.train --module alphabet_sort \
    --config $CFG --metrics.no-enable-wandb --async-loop.num-training-steps $STEPS "$@" \
    > rl_eval/logs/tier1_cost_${tag}.log 2>&1
  echo "===== $tag DONE $(date -u +%H:%M:%S) rc=$? ====="
}
# ON = config default (batch_invariant + deterministic already set)
run ON
# OFF = disable batch invariance + determinism on both trainer and generator
run OFF --trainer.debug.no-batch-invariant --generator.debug.no-batch-invariant \
        --trainer.debug.no-deterministic
echo "ALL DONE $(date -u)"
