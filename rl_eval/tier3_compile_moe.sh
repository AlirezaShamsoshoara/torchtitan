#!/bin/bash
source rl_eval/activate_env.sh
export no_proxy="localhost,127.0.0.1,.internalfb.com,.facebook.com,.fbcdn.net,.tfbnw.net,.fb.com,.fbinfra.net"; export NO_PROXY="$no_proxy"

# 1. torch.compile ON vs OFF (base 0.6b varlen, 6 steps)
for C in on off; do
  echo "===== compile=$C START $(date -u +%H:%M:%S) ====="
  rm -rf outputs/rl/checkpoint 2>/dev/null
  FLAG=""; [ "$C" = "off" ] && FLAG="--compile.no-enable"
  python -m torchtitan.experiments.rl.train --module alphabet_sort \
    --config rl_grpo_qwen3_0_6b_varlen --metrics.no-enable-wandb \
    --async-loop.num-training-steps 6 $FLAG \
    > rl_eval/logs/tier3_compile_${C}.log 2>&1
  echo "===== compile=$C DONE $(date -u +%H:%M:%S) rc=$? ====="
done

# 2. MoE debug (HybridEP, debugmodel_moe, EP=4, 8 GPU)
echo "===== MoE debug_varlen START $(date -u +%H:%M:%S) ====="
rm -rf outputs/rl/checkpoint 2>/dev/null
python -m torchtitan.experiments.rl.train --module alphabet_sort \
  --config rl_grpo_qwen3_moe_debug_varlen --metrics.no-enable-wandb \
  --async-loop.num-training-steps 5 \
  > rl_eval/logs/tier3_moe_varlen.log 2>&1
echo "===== MoE debug_varlen DONE $(date -u +%H:%M:%S) rc=$? ====="

# 3. MoE DeepEP comm backend
echo "===== MoE deepep START $(date -u +%H:%M:%S) ====="
rm -rf outputs/rl/checkpoint 2>/dev/null
python -m torchtitan.experiments.rl.train --module alphabet_sort \
  --config rl_grpo_qwen3_moe_debug_deepep --metrics.no-enable-wandb \
  --async-loop.num-training-steps 5 \
  > rl_eval/logs/tier3_moe_deepep.log 2>&1
echo "===== MoE deepep DONE $(date -u +%H:%M:%S) rc=$? ====="
echo "COMPILE+MOE DONE $(date -u)"
