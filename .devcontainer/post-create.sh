#!/usr/bin/env bash
set -euo pipefail

PYTORCH_VERSION="${PYTORCH_VERSION:-2.11.*}"
TORCHCODEC_VERSION="${TORCHCODEC_VERSION:-0.15.*}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

python -m pip install --upgrade pip setuptools wheel

pytorch_packages=(
  "torch==${PYTORCH_VERSION}"
  "torchcodec==${TORCHCODEC_VERSION}"
)

python -m pip install \
  --index-url "${PYTORCH_INDEX_URL}" \
  "${pytorch_packages[@]}"

python -m pip install -r requirements.txt
python -m pip install pytest

python - <<'PY'
from importlib.metadata import version

import librosa
import numpy
import soxr
import torch
from torchcodec.decoders import AudioDecoder

print("Development environment ready:")
print(f"  torch:      {torch.__version__}")
print(f"  torchcodec: {version('torchcodec')}")
print(f"  librosa:    {librosa.__version__}")
print(f"  numpy:      {numpy.__version__}")
print(f"  soxr:       {soxr.__version__}")
print(f"  decoder:    {AudioDecoder.__name__}")
PY

ffmpeg -version | sed -n '1p'
