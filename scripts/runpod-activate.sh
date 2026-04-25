#!/usr/bin/env bash
# activate the phloem environment on a RunPod pod.

echo "=== configuring env vars ==="

# ensure uv on path
source "$HOME/.local/bin/env"

# HF model cache on network volume (large, persists across pods)
export HF_HOME=/workspace/huggingface

# venv on container disk (fast IOPS for python imports)
export UV_PROJECT_ENVIRONMENT=/root/phloem-venv

# wandb
export WANDB_ENTITY=vivshaw

echo "ready. HF_HOME=$HF_HOME, venv=$UV_PROJECT_ENVIRONMENT"
