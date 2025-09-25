#!/bin/bash

# Two-stage training script for diffusion action head grafting
# 两阶段训练脚本：将MLP头替换为cross-attention + self-attention + MLP结构

set -e

# Configuration
CHECKPOINT_PATH="checkpoints/pusht.ckpt"  # Path to your pretrained model
DATASET_PATH="data/pusht/"  # Path to your dataset
OUTPUT_DIR="outputs/graft_head_training"
NUM_SAMPLES=8000  # Number of samples for Stage 1
BATCH_SIZE=32
DEVICE="cuda:0"

echo "=========================================="
echo "Two-Stage Diffusion Head Grafting Training"
echo "=========================================="

# Create output directories
mkdir -p ${OUTPUT_DIR}/stage1/cached_activations
mkdir -p ${OUTPUT_DIR}/stage1/init_weights
mkdir -p ${OUTPUT_DIR}/stage2/checkpoints

echo "Stage 1: Caching teacher activations..."
echo "----------------------------------------"

# Stage 1: Cache teacher activations
CUDA_VISIBLE_DEVICES=0 python unified_video_action/tools/cache_activation.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --dataset ${DATASET_PATH} \
    --out ${OUTPUT_DIR}/stage1/cached_activations \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE}

echo ""
echo "Stage 1: Distilling individual operators..."
echo "------------------------------------------"

# Stage 1: Distill cross-attention operator
echo "Distilling cross-attention operator..."
CUDA_VISIBLE_DEVICES=0 python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer cross_attn \
    --loss l1 \
    --out ${OUTPUT_DIR}/stage1/init_weights/cross_attn.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64

# Stage 1: Distill self-attention operator
echo "Distilling self-attention operator..."
CUDA_VISIBLE_DEVICES=0 python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer self_attn \
    --loss l1 \
    --out ${OUTPUT_DIR}/stage1/init_weights/self_attn.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64

# Stage 1: Distill MLP operator
echo "Distilling MLP operator..."
CUDA_VISIBLE_DEVICES=0 python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer mlp \
    --loss l2 \
    --out ${OUTPUT_DIR}/stage1/init_weights/mlp.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64

echo ""
echo "Stage 2: End-to-end fine-tuning..."
echo "--------------------------------"

# Stage 2: End-to-end fine-tuning with distilled weights
accelerate launch --num_processes=1 train.py \
    --config-name=train_diffusion_graft_workspace \
    model.policy.action_model_params.head_init_paths.cross_attn=${OUTPUT_DIR}/stage1/init_weights/cross_attn.pt \
    model.policy.action_model_params.head_init_paths.self_attn=${OUTPUT_DIR}/stage1/init_weights/self_attn.pt \
    model.policy.action_model_params.head_init_paths.mlp=${OUTPUT_DIR}/stage1/init_weights/mlp.pt \
    model.policy.finetune.freeze_encoder=true \
    training.steps=50000 \
    training.data_fraction=0.10 \
    model.policy.optimizer.learning_rate=1e-4 \
    hydra.run.dir=${OUTPUT_DIR}/stage2/checkpoints

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/stage2/checkpoints"
echo "=========================================="
