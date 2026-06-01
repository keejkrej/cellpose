#!/usr/bin/env bash
# Local AMD/ROCm setup: sync cellpose deps but install PyTorch from ROCm nightlies
# instead of the CUDA wheels pinned in uv.lock. Safe to commit; only runs when you invoke it.
set -euo pipefail
cd "$(dirname "$0")"

ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/nightly/rocm7.2}"

echo "Syncing project (skipping CUDA torch/torchvision)..."
uv sync --no-install-package torch --no-install-package torchvision

echo "Installing ROCm PyTorch from ${ROCM_INDEX}..."
uv pip install --pre torch torchvision torchaudio --index-url "${ROCM_INDEX}"

python -c "import torch; print('torch', torch.__version__, 'hip', getattr(torch.version, 'hip', None), 'cuda_available', torch.cuda.is_available())"
