#!/bin/bash
# Tier 3: async off-policy sweep. Same config, vary max_offpolicy_steps in {0,1,3}.
source rl_eval/activate_env.sh
export no_proxy="localhost,127.0.0.1,.internalfb.com,.facebook.com,.fbcdn.net,.tfbnw.net,.fb.com,.fbinfra.net"; export NO_PROXY="$no_proxy"
STEPS=8
for OPS in 0 1 3; do
  echo "===== max_offpolicy_steps=$OPS START $(date -u +%H:%M:%S) ====="
  rm -rf outputs/rl/checkpoint 2>/dev/null
  python -m torchtitan.experiments.rl.train --module alphabet_sort \
    --config rl_grpo_qwen3_0_6b_varlen --metrics.no-enable-wandb \
    --async-loop.num-training-steps $STEPS --async-loop.max-offpolicy-steps $OPS \
    > rl_eval/logs/tier3_offpolicy_${OPS}.log 2>&1
  echo "===== max_offpolicy_steps=$OPS DONE $(date -u +%H:%M:%S) rc=$? ====="
done
echo "OFFPOLICY SWEEP DONE $(date -u)"
