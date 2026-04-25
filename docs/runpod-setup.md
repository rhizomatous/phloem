# RunPod setup

## prerequisites (one-time)

- **RunPod account** with credits added
- **SSH public key** added to RunPod for terminal access

## 1. create a network volume

network volumes persist independently of pods. you can spin pods up & down without losing data on network vols. pods can only attach volumes in the **same region**, so choose wisely.

- region: e.g. `US-CA-2` (any region with availability for the GPU you crave)
- size: **~100-200 GB** (room for HF model cache, SAE checkpoints, multiple runs... don't skimp)

## 2. create a pod

- **filter to your network volume's region**
- pick your GPU
- **template:** RunPod PyTorch 2.x (comes with CUDA, Python, git)
- **container disk:** 50 GB (venv + uv cache live here for fast imports)
- **network volume:** attach the one from step 1, mount at `/workspace`
- deploy

## 3. connect

```bash
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519
```

## 4. pod setup

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# triton needs python dev headers to compile its CUDA kernels
apt-get update && apt-get install -y python3.10-dev

# HF model cache on network volume
export HF_HOME=/workspace/huggingface

# venv on container disk (otherwise slow as molasses due to IOPS)
export UV_PROJECT_ENVIRONMENT=/root/phloem-venv

# set up the lab
cd /workspace
git clone https://github.com/rhizomatous/phloem.git
cd phloem

# install deps
uv sync

# add your HF token
cat > .env << 'EOF'
HF_TOKEN=hf_your_token_here
EOF
```

# other useful tidbits

## new pod with same volume

```bash
apt-get update && apt-get install -y python3.10-dev

uv sync

source $HOME/.local/bin/env

export HF_HOME=/workspace/huggingface
export UV_PROJECT_ENVIRONMENT=/root/phloem-venv
```

## retrieving trained SAEs locally

checkpoints land in `models/gemma-4-e2b/checkpoints/<run-name>/`. each hookpoint gets its own subdirectory with `sae.safetensors` and `cfg.json`. can pull with scp:

```bash
# from your local machine
scp -P <pod-port> -r root@<pod-ip>:/workspace/phloem/models/gemma-4-e2b/checkpoints/<run-name> ./
```

or, push to HuggingFace Hub from the pod.

## full run

```
uv run python models/gemma-4-e2b/train.py \
    --max-tokens 8000000000 \
    --layers all \
    --seq-len 1024 \
    --batch-size 16 \
    --expansion-factor 32 \
    --wandb \
    --run-name all-layers-32x-8B

uv run python models/gemma-4-e2b/eval.py models/gemma-4-e2b/checkpoints/all-layers-32x-8B

uv run python models/gemma-4-e2b/max_activations.py \
    models/gemma-4-e2b/checkpoints/all-layers-32x-8B \
    --max-tokens 10000000 \
    --layers all
```