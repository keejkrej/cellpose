#!/usr/bin/env bash
# Fast ROCm setup: sync from uv.lock but skip the CUDA torch stack, then pip ROCm torch.
# Do not run plain `uv sync` after this — it will re-pull CUDA. Re-run this script instead.
set -euo pipefail
cd "$(dirname "$0")"

ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/nightly/rocm7.2}"

# CUDA-only packages pulled in by cu130 torch in uv.lock (not needed for ROCm).
SKIP=(
  torch torchvision torchaudio triton
  cuda-bindings cuda-pathfinder cuda-toolkit
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13
  nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx
)

args=(uv sync --frozen)
for pkg in "${SKIP[@]}"; do
  args+=(--no-install-package "$pkg")
done

echo "Syncing cellpose deps (skipping CUDA torch stack)..."
"${args[@]}"

echo "Installing ROCm PyTorch from ${ROCM_INDEX}..."
# uv 0.11 can fail extracting huge ROCm torch wheels (zip64); pip handles them fine.
uv pip install -q pip
.venv/bin/pip install --pre torch torchvision torchaudio --index-url "${ROCM_INDEX}"

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'hip', getattr(torch.version, 'hip', None), 'cuda_available', torch.cuda.is_available())"
