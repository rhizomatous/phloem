#!/usr/bin/env bash
# first-time setup on a fresh RunPod pod with a network volume at /workspace.
set -euo pipefail

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

source scripts/runpod-activate.sh

echo "=== installing python dev headers (triton needs these) ==="
apt-get update && apt-get install -y python3.10-dev

echo "=== installing deps ==="
uv sync

echo "=== wandb login ==="
uv run wandb login

echo "=== setting up .env ==="
if [ ! -f .env ]; then
    echo "HF_TOKEN=hf_your_token_here"
    echo "!! create .env and add your real HF token !!"
else
    echo ".env already exists, skipping"
fi

echo ""
echo "=== done! ==="
