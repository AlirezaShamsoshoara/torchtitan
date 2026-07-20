#!/bin/bash
# TitanRL env build — follows upstream RL README, isolated uv venv (venv_titanrl).
# Does NOT touch system python / conda / default profile.
set -uo pipefail
cd /home/alisol/projects/torchtitan
ROOT="$PWD"
VENV="$ROOT/venv_titanrl"
step() { echo -e "\n\n========== STEP: $* ========== ($(date -u +%H:%M:%S))" ; }

echo "BUILD START $(date -u)"

step "0. Create isolated uv venv (python 3.12) named venv_titanrl"
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
echo "which python: $(which python)"; python --version

step "1. Monarch, TorchStore, Renderers, helpers"
uv pip install torchmonarch
uv pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install pygtrie portpicker
uv pip install "git+https://github.com/PrimeIntellect-ai/renderers.git@main"

step "2. Flash Attention 3 (cu130)"
uv pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130

step "3. batch_invariant_ops (Tier 1 bitwise/batch-invariant)"
uv pip install --no-deps "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"

step "4. PyTorch nightly + vLLM prebuilt + torchcomms nightly (cu130, date-aligned)"
uv pip install torch vllm torchcomms --pre \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --index-strategy unsafe-best-match
echo "--- torch pinned to: $(python -c 'import torch;print(torch.__version__)' 2>&1)"
# torchvision is NOT in the RL README recipe, but vLLM nightly's kernel_warmup path
# (minimax_m3 warmup) imports it unconditionally -> smoke test crashes without it.
# Install --no-deps so it can't disturb the torch pin. GAP flagged for Tier 5.
uv pip install torchvision --pre \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --index-strategy unsafe-best-match --no-deps

step "5a. torchtitan runtime deps WITHOUT disturbing nightly torch"
# torch-dependent dep installed with --no-deps (torch already satisfies it)
uv pip install --no-deps torchdata
# the rest carry no torch dependency
uv pip install "datasets>=3.6.0,<4.8.0" tokenizers safetensors tyro tensorboard wandb einops pillow "spmd_types==0.2.1"

step "5b. Register torchtitan package (editable, no-deps) + will also export PYTHONPATH at runtime"
uv pip install -e . --no-deps

step "5c. GUARD: ensure nightly torch/vllm not clobbered by dep resolution"
TORCH_V=$(python -c 'import torch;print(torch.__version__)' 2>&1)
echo "torch after deps: $TORCH_V"
if [[ "$TORCH_V" != *"dev"* ]]; then
  echo "!!! torch nightly was clobbered ($TORCH_V) — re-pinning"
  uv pip install torch vllm torchcomms --pre \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --index-strategy unsafe-best-match --reinstall-package torch
fi

step "6. Version report"
python - << 'PYEOF'
import importlib.metadata as m
pkgs = ['torch','vllm','torchcomms','torchmonarch','monarch','torchstore',
        'flash-attn-3','flash_attn_3','batch_invariant_ops','renderers',
        'pygtrie','portpicker','torchtitan','triton','torchdata','datasets',
        'spmd_types','tyro','wandb','tensorboard']
for p in pkgs:
    try: print(f'{p:22s} {m.version(p)}')
    except Exception: print(f'{p:22s} NOT FOUND')
PYEOF

echo -e "\nBUILD DONE $(date -u)"
