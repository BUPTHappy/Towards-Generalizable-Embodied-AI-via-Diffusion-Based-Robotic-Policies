#!/bin/bash
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

# 改进的微调训练脚本
LOG_FILE="improved_finetune_$(date +%Y%m%d_%H%M%S).log"
echo "Starting improved finetune strategy..."
echo "Strategy: Conservative training with proper initialization"
echo "All output will be logged to: $LOG_FILE"

# 检查是否在screen中运行
if [ -z "$STY" ]; then
    echo "Not running in screen. Starting screen session..."
    echo "Creating screen session 'improved_finetune'..."
    
    # 创建screen会话并运行脚本
    screen -dmS improved_finetune bash -c "
        echo 'Screen session started at: \$(date)'
        echo 'Working directory: \$(pwd)'
        echo 'Starting improved finetune training...'
        bash $0
        echo 'Training completed at: \$(date)'
        echo 'Press any key to exit...'
        read -n 1
    "
    
    echo "✅ Training started in screen session 'improved_finetune'"
    echo ""
    echo "📋 Useful commands:"
    echo "  • Attach to session:    screen -r improved_finetune"
    echo "  • Detach from session:  Ctrl+A, then D"
    echo "  • List all sessions:    screen -ls"
    echo "  • Kill session:         screen -S improved_finetune -X quit"
    echo ""
    echo "📁 Log file: $LOG_FILE"
    echo "📊 Monitor progress: tail -f $LOG_FILE"
    echo ""
    echo "The training is now running in the background!"
    exit 0
fi

echo "Running in screen session: $STY"

# 设置日志输出
exec > >(tee -a $LOG_FILE) 2>&1

echo 'Starting improved finetune training...'
echo "Log file: $LOG_FILE"
echo "Timestamp: $(date)"

# 设置错误处理
set -e
trap 'echo "Error occurred at line $LINENO. Check log file: $LOG_FILE"' ERR

# 创建checkpoints目录
mkdir -p checkpoints

# ===========================================
# 步骤1：设置改进的训练策略
# ===========================================
echo '=== STEP 1: Setting up improved training strategy ==='
echo "Setup started at: $(date)"

/home/shane/miniforge3/condabin/conda run -n uva python improved_finetune_strategy.py

echo 'Improved strategy setup completed!'
echo "Setup finished at: $(date)"

# ===========================================
# 步骤2：阶段1训练 - Local Attention Only
# ===========================================
echo '=== STEP 2: Phase 1 - Local Attention Only ==='
echo "Phase 1 started at: $(date)"

# 阶段1训练 - 更保守的设置
echo 'Starting Phase 1 training with improved settings...'
/home/shane/miniforge3/condabin/conda run -n uva accelerate launch --num_processes=1 train.py \
    --config-dir=unified_video_action/config \
    --config-name=uva_pusht \
    model.policy.autoregressive_model_params.pretrained_model_path='checkpoints/improved_phase1_local_only.ckpt' \
    model.policy.action_model_params.predict_action=True \
    training.num_epochs=50 \
    model.policy.optimizer.learning_rate=5e-5 \
    dataloader.batch_size=8 \
    logging.mode=offline \
    hydra.run.dir='checkpoints/improved_phase1_local_attention'

echo 'Phase 1 training completed!'
echo "Phase 1 finished at: $(date)"

# 评估阶段1性能
echo 'Evaluating Phase 1 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python evaluate_performance.py \
    --checkpoint='checkpoints/improved_phase1_local_attention/checkpoints/topk.ckpt' \
    --phase='Improved_Phase_1_Local_Attention'

# 性能检查
echo 'Checking Phase 1 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python check_performance_degradation.py \
    --current-checkpoint='checkpoints/improved_phase1_local_attention/checkpoints/topk.ckpt' \
    --baseline-checkpoint='checkpoints/pusht.ckpt' \
    --phase-name='Phase_1' \
    --threshold=0.1

if [ $? -ne 0 ]; then
    echo "❌ Phase 1 performance check failed!"
    echo "Consider adjusting learning rate or initialization"
    echo "Continuing to Phase 2 with caution..."
fi

# ===========================================
# 步骤3：阶段2训练 - Local Attention + Partial Encoder
# ===========================================
echo '=== STEP 3: Phase 2 - Local Attention + Partial Encoder ==='
echo "Phase 2 started at: $(date)"

# 加载阶段1的结果并设置阶段2
cat > setup_improved_phase2.py << 'EOF'
import torch
import sys
import os
sys.path.append("/home/shane/code_b/unified_video_action")

import hydra
from omegaconf import OmegaConf
from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy

# 加载配置
from hydra import initialize, compose
with initialize(config_path="unified_video_action/config"):
    cfg = compose(config_name="uva_pusht")

OmegaConf.resolve(cfg)
policy = UnifiedVideoActionPolicy(**cfg.model.policy, normalizer_type=cfg.task.dataset.normalizer_type, task_name=cfg.task.name)

# 加载阶段1的checkpoint
checkpoint_path = "checkpoints/improved_phase1_local_attention/checkpoints/topk.ckpt"
if not os.path.exists(checkpoint_path):
    print(f"ERROR: Phase 1 checkpoint not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location="cpu")
if "state_dict" in checkpoint:
    policy.load_state_dict(checkpoint["state_dict"])
elif "state_dicts" in checkpoint and "ema_model" in checkpoint["state_dicts"]:
    ema_state_dict = checkpoint["state_dicts"]["ema_model"]
    model_state_dict = {k[6:]: v for k, v in ema_state_dict.items() if k.startswith("model.")}
    policy.load_state_dict(model_state_dict, strict=False)
else:
    policy.load_state_dict(checkpoint, strict=False)

print(f"Successfully loaded Phase 1 checkpoint from: {checkpoint_path}")

# 冻结所有参数
for param in policy.model.parameters():
    param.requires_grad = False

# 解冻参数：local attention + 少量encoder层
phase2_params = []
for name, param in policy.model.named_parameters():
    should_unfreeze = False
    
    # Local attention参数（保持解冻）
    if any(keyword in name for keyword in ['local_causal_blocks', 'feature_fusion', 'gate']):
        should_unfreeze = True
    
    # 解冻encoder的最后2层（layer 10-11）
    elif ('encoder_blocks' in name and 
          any(f'.{i}.' in name for i in range(10, 12))):
        should_unfreeze = True
    
    # 解冻encoder norm
    elif 'encoder_norm' in name:
        should_unfreeze = True
    
    if should_unfreeze:
        param.requires_grad = True
        phase2_params.append(name)

print(f"Phase 2 unfrozen parameters: {len(phase2_params)}")

# 保存模型
checkpoint = {
    "state_dicts": {
        "ema_model": {f"model.{k}": v for k, v in policy.state_dict().items()}
    }
}
torch.save(checkpoint, "checkpoints/improved_phase2_partial.ckpt")
print("Phase 2 setup completed")
EOF

/home/shane/miniforge3/condabin/conda run -n uva python setup_improved_phase2.py

# 阶段2训练
echo 'Starting Phase 2 training...'
/home/shane/miniforge3/condabin/conda run -n uva accelerate launch --num_processes=1 train.py \
    --config-dir=unified_video_action/config \
    --config-name=uva_pusht \
    model.policy.autoregressive_model_params.pretrained_model_path='checkpoints/improved_phase2_partial.ckpt' \
    model.policy.action_model_params.predict_action=True \
    training.num_epochs=30 \
    model.policy.optimizer.learning_rate=3e-5 \
    dataloader.batch_size=8 \
    logging.mode=offline \
    hydra.run.dir='checkpoints/improved_phase2_partial_finetune'

echo 'Phase 2 training completed!'

# 评估阶段2性能
echo 'Evaluating Phase 2 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python evaluate_performance.py \
    --checkpoint='checkpoints/improved_phase2_partial_finetune/checkpoints/topk.ckpt' \
    --phase='Improved_Phase_2_Partial_Encoder'

# ===========================================
# 步骤4：阶段3训练 - Local Attention + Extended Encoder
# ===========================================
echo '=== STEP 4: Phase 3 - Local Attention + Extended Encoder ==='
echo "Phase 3 started at: $(date)"

# 加载阶段2的结果并设置阶段3
cat > setup_improved_phase3.py << 'EOF'
import torch
import sys
import os
sys.path.append("/home/shane/code_b/unified_video_action")

import hydra
from omegaconf import OmegaConf
from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy

# 加载配置
from hydra import initialize, compose
with initialize(config_path="unified_video_action/config"):
    cfg = compose(config_name="uva_pusht")

OmegaConf.resolve(cfg)
policy = UnifiedVideoActionPolicy(**cfg.model.policy, normalizer_type=cfg.task.dataset.normalizer_type, task_name=cfg.task.name)

# 加载阶段2的checkpoint
checkpoint_path = "checkpoints/improved_phase2_partial_finetune/checkpoints/topk.ckpt"
if not os.path.exists(checkpoint_path):
    print(f"ERROR: Phase 2 checkpoint not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location="cpu")
if "state_dict" in checkpoint:
    policy.load_state_dict(checkpoint["state_dict"])
elif "state_dicts" in checkpoint and "ema_model" in checkpoint["state_dicts"]:
    ema_state_dict = checkpoint["state_dicts"]["ema_model"]
    model_state_dict = {k[6:]: v for k, v in ema_state_dict.items() if k.startswith("model.")}
    policy.load_state_dict(model_state_dict, strict=False)
else:
    policy.load_state_dict(checkpoint, strict=False)

print(f"Successfully loaded Phase 2 checkpoint from: {checkpoint_path}")

# 冻结所有参数
for param in policy.model.parameters():
    param.requires_grad = False

# 解冻参数：local attention + 更多encoder层
phase3_params = []
for name, param in policy.model.named_parameters():
    should_unfreeze = False
    
    # Local attention参数（保持解冻）
    if any(keyword in name for keyword in ['local_causal_blocks', 'feature_fusion', 'gate']):
        should_unfreeze = True
    
    # 解冻encoder的后半部分（layer 6-11）
    elif ('encoder_blocks' in name and 
          any(f'.{i}.' in name for i in range(6, 12))):
        should_unfreeze = True
    
    # 解冻encoder norm
    elif 'encoder_norm' in name:
        should_unfreeze = True
    
    # 解冻decoder的前几层
    elif ('decoder_blocks' in name and 
          any(f'.{i}.' in name for i in range(0, 4))):
        should_unfreeze = True
    
    if should_unfreeze:
        param.requires_grad = True
        phase3_params.append(name)

print(f"Phase 3 unfrozen parameters: {len(phase3_params)}")

# 保存模型
checkpoint = {
    "state_dicts": {
        "ema_model": {f"model.{k}": v for k, v in policy.state_dict().items()}
    }
}
torch.save(checkpoint, "checkpoints/improved_phase3_extended.ckpt")
print("Phase 3 setup completed")
EOF

/home/shane/miniforge3/condabin/conda run -n uva python setup_improved_phase3.py

# 阶段3训练
echo 'Starting Phase 3 training...'
/home/shane/miniforge3/condabin/conda run -n uva accelerate launch --num_processes=1 train.py \
    --config-dir=unified_video_action/config \
    --config-name=uva_pusht \
    model.policy.autoregressive_model_params.pretrained_model_path='checkpoints/improved_phase3_extended.ckpt' \
    model.policy.action_model_params.predict_action=True \
    training.num_epochs=20 \
    model.policy.optimizer.learning_rate=1e-5 \
    dataloader.batch_size=8 \
    logging.mode=offline \
    hydra.run.dir='checkpoints/improved_phase3_extended_finetune'

echo 'Phase 3 training completed!'

# 评估阶段3性能
echo 'Evaluating Phase 3 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python evaluate_performance.py \
    --checkpoint='checkpoints/improved_phase3_extended_finetune/checkpoints/topk.ckpt' \
    --phase='Improved_Phase_3_Extended_Encoder'

echo '=== IMPROVED FINETUNE TRAINING COMPLETED! ==='
echo 'Final model saved in: checkpoints/improved_phase3_extended_finetune/checkpoints/topk.ckpt'
echo ''
echo '=== TRAINING SUMMARY ==='
echo 'Strategy: Conservative progressive training with proper initialization'
echo 'Phase 1: Local attention only (50 epochs, 5e-5 lr)'
echo 'Phase 2: + Encoder layers 10-11 (30 epochs, 3e-5 lr)'  
echo 'Phase 3: + Encoder layers 6-11 + Decoder layers 0-3 (20 epochs, 1e-5 lr)'
echo ''
echo 'Key improvements:'
echo '- Proper initialization of local attention parameters'
echo '- Increased training epochs for better convergence'
echo '- Lower learning rates to prevent instability'
echo '- Conservative parameter unfreezing strategy'
echo '- Better performance monitoring'

# 生成性能比较报告
echo 'Generating performance comparison report...'
/home/shane/miniforge3/condabin/conda run -n uva python compare_performance.py \
    --checkpoints-dir='checkpoints' \
    --output-report='improved_finetune_report.txt'

echo "Improved finetune training completed"
echo "All output is logged to: $LOG_FILE"
echo "To view real-time progress: tail -f $LOG_FILE"
