#!/bin/bash
# Run all training phases sequentially on remote instance

set -e

cd /workspace/codeagent-rwkv7

echo "========================================="
echo "CodeAgent-RWKV Multi-Phase Training"
echo "========================================="

# Phase 1: 4096 context
echo ""
echo ">>> PHASE 1: Context 4096 (Foundation)"
echo "========================================="
python train_phase.py --phase 1 2>&1 | tee logs/phase1.log

# Phase 2: 16384 context
echo ""
echo ">>> PHASE 2: Context 16384 (Extension)"
echo "========================================="
python train_phase.py --phase 2 2>&1 | tee logs/phase2.log

# Phase 3: 65536 context
echo ""
echo ">>> PHASE 3: Context 65536 (Extension)"
echo "========================================="
python train_phase.py --phase 3 2>&1 | tee logs/phase3.log

# Phase 4: 131072 context
echo ""
echo ">>> PHASE 4: Context 131072 (Final)"
echo "========================================="
python train_phase.py --phase 4 2>&1 | tee logs/phase4.log

echo ""
echo "========================================="
echo "ALL PHASES COMPLETE!"
echo "Final model: checkpoints/phase4/final"
echo "========================================="
