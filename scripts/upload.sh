#!/bin/bash
# Upload local code to Vast.ai instance and start training

set -e

INSTANCE_HOST="ssh8.vast.ai"
INSTANCE_PORT="16528"
REMOTE_DIR="/workspace/codeagent-rwkv7"

echo "Uploading training code to Vast.ai instance..."

# Create remote directory
ssh -p $INSTANCE_PORT root@$INSTANCE_HOST "mkdir -p $REMOTE_DIR"

# Sync files
rsync -avz --progress \
    -e "ssh -p $INSTANCE_PORT" \
    --exclude='.venv' \
    --exclude='checkpoints' \
    --exclude='logs' \
    --exclude='data' \
    --exclude='.git' \
    ./ root@$INSTANCE_HOST:$REMOTE_DIR/

echo "Upload complete!"
