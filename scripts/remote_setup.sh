#!/bin/bash
# Remote setup script for CodeAgent-RWKV training on Vast.ai

set -e

echo "========================================="
echo "CodeAgent-RWKV Remote Setup"
echo "========================================="

# Update system
echo "[1/6] Updating system packages..."
apt-get update -qq
apt-get install -y -qq git wget curl rsync htop nvtop screen tmux

# Install Python dependencies
echo "[2/6] Installing Python packages..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers datasets accelerate huggingface-hub
pip install -q bitsandbytes peft wandb pyyaml tqdm
pip install -q sentencepiece protobuf

# Login to HF and WandB if keys provided
if [ -n "$HF_TOKEN" ]; then
    echo "[3/6] Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "[3/6] No HF_TOKEN provided, datasets may fail if gated"
fi

if [ -n "$WANDB_API_KEY" ]; then
    echo "[4/6] Logging into Weights & Biases..."
    wandb login "$WANDB_API_KEY"
else
    echo "[4/6] No WANDB_API_KEY provided, logging to files only"
fi

# Create workspace
echo "[5/6] Setting up workspace..."
mkdir -p /workspace/codeagent-rwkv7
cd /workspace/codeagent-rwkv7
mkdir -p checkpoints logs data

# Set environment variables for training
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

echo "[6/6] Setup complete!"
echo "========================================="
echo "Ready for training"
echo "========================================="
