#!/bin/bash
# Run this on a fresh g5.xlarge/g5.2xlarge with Deep Learning AMI (Ubuntu)
# AMI: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)"
# Region: us-east-1 (cheapest for g5)

set -euo pipefail

echo "=== Verifying GPU ==="
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

echo "=== Installing project dependencies ==="
cd ~
if [ ! -d "fedlora-poison" ]; then
    git clone <YOUR_REPO_URL> fedlora-poison
fi
cd fedlora-poison

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Create venv and install
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

echo "=== Smoke test ==="
python -c "
import torch, transformers, peft, flwr
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'transformers {transformers.__version__}')
print(f'peft {peft.__version__}')
print(f'flwr {flwr.__version__}')
print('All imports OK')
"

echo "=== Setup complete ==="
echo "Remember: stop your instance when done! GPU hours add up."
