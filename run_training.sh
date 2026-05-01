#!/bin/bash
set -e
cd /workspace/codeagent-rwkv7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "========================================"
echo "CodeAgent-RWKV7 Training Starting"
echo "========================================"

for phase in 1 2 3 4; do
    echo ""
    echo ">>> Starting Phase $phase"
    echo "========================================"
    python3 train_phase.py --phase $phase 2>&1 | tee logs/phase${phase}.log
    echo ">>> Phase $phase complete"
done

echo ""
echo "========================================"
echo "ALL PHASES COMPLETE"
echo "Final model: checkpoints/phase4/final"
echo "========================================"
