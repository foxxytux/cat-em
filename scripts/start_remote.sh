#!/bin/bash
# Start training on remote instance via SSH

INSTANCE_HOST="ssh8.vast.ai"
INSTANCE_PORT="16528"
REMOTE_DIR="/workspace/codeagent-rwkv7"

echo "Starting remote training..."

# Run setup first
ssh -p $INSTANCE_PORT root@$INSTANCE_HOST "cd $REMOTE_DIR && bash scripts/remote_setup.sh"

# Start training in a tmux session so it persists
ssh -p $INSTANCE_PORT root@$INSTANCE_HOST "
    cd $REMOTE_DIR
    tmux new-session -d -s rwkv-training 'bash scripts/run_all_phases.sh'
    echo 'Training started in tmux session: rwkv-training'
    echo 'Attach with: tmux attach -t rwkv-training'
"

echo ""
echo "Training is running remotely!"
echo "Monitor with: ssh -p $INSTANCE_PORT root@$INSTANCE_HOST 'tmux attach -t rwkv-training'"
