#!/usr/bin/env bash
# Install cellpose into the *active* conda/mamba env with ROCm PyTorch (no uv).
#
#   mamba create -n cellpose python=3.12 pip -y
#   mamba activate cellpose
#   ./setup-rocm.sh
#
# Then: cellpose          # GUI (no image args)
#       cellpose --use_gpu --dir /path/to/images
set -euo pipefail
cd "$(dirname "$0")"

ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/nightly/rocm7.2}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "error: activate a conda/mamba environment first (CONDA_PREFIX is unset)." >&2
  echo "  mamba create -n cellpose python=3.12 pip -y && mamba activate cellpose" >&2
  exit 1
fi

if ! command -v python >/dev/null || ! command -v pip >/dev/null; then
  echo "error: python and pip must be on PATH in the active environment." >&2
  exit 1
fi

echo "Environment: ${CONDA_PREFIX}"
echo "Python: $(python -V)"

echo "Installing ROCm PyTorch from ${ROCM_INDEX}..."
python -m pip install --upgrade pip
python -m pip install --pre torch torchvision torchaudio --index-url "${ROCM_INDEX}"

echo "Installing cellpose dependencies (excluding torch/torchvision)..."
mapfile -t DEPS < <(python - <<'PY'
import re
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text())
deps = []
for spec in data["project"]["dependencies"]:
    name = re.split(r"[\s[<>=!;@]", spec, maxsplit=1)[0].lower()
    if name in ("torch", "torchvision", "torchaudio"):
        continue
    deps.append(spec)
print("\n".join(deps))
PY
)
python -m pip install "${DEPS[@]}"

echo "Installing cellpose (editable)..."
python -m pip install "hatchling==1.27.0" "hatch-vcs>=0.4" "pathspec<0.12"
python -m pip install --no-deps -e .

python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
from cellpose import version, version_str
print(version_str)
print("cellpose CLI:", end=" ")
import shutil
print(shutil.which("cellpose") or "(run: python -m cellpose)")
PY
