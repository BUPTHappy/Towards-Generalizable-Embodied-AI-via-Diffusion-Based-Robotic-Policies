#!/bin/bash

# Quick test script for Stage 2 only (assuming Stage 1 is completed)
# 快速测试脚本：仅运行Stage 2（假设Stage 1已完成）

set -e

# Configuration - adjust these paths
INIT_WEIGHTS_DIR="outputs/graft_head_training/stage1/init_weights"
OUTPUT_DIR="outputs/graft_head_test"
CHECKPOINT_PATH="checkpoints/pusht.ckpt"

echo "=========================================="
echo "Quick Test: Stage 2 Fine-tuning"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Running Stage 2 with distilled weights..."
echo "----------------------------------------"

# Stage 2: End-to-end fine-tuning with distilled weights
accelerate launch --num_processes=1 train.py \
    --config-name=train_diffusion_graft_workspace \
    model.policy.action_model_params.head_init_paths.cross_attn=${INIT_WEIGHTS_DIR}/cross_attn.pt \
    model.policy.action_model_params.head_init_paths.self_attn=${INIT_WEIGHTS_DIR}/self_attn.pt \
    model.policy.action_model_params.head_init_paths.mlp=${INIT_WEIGHTS_DIR}/mlp.pt \
    model.policy.finetune.freeze_encoder=true \
    training.steps=10000 \
    training.data_fraction=0.05 \
    model.policy.optimizer.learning_rate=1e-4 \
    hydra.run.dir=${OUTPUT_DIR}

echo ""
echo "Quick test completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=========================================="
