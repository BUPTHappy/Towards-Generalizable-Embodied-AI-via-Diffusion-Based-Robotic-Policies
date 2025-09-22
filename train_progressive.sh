#!/bin/bash
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# 设置日志文件
LOG_FILE="progressive_training_$(date +%Y%m%d_%H%M%S).log"
echo "Starting progressive training strategy..."
echo "Phase 1: Only local attention parameters"
echo "Phase 2: Partial unfreezing for fine-tuning"
echo "Phase 3: More parameters unfrozen for full fine-tuning"
echo "All output will be logged to: $LOG_FILE"

# 检查是否在screen中运行
if [ -z "$STY" ]; then
    echo "Not running in screen. Starting screen session..."
    screen -dmS progressive_training bash -c "bash $0; exec bash"
    echo "Training started in screen session 'progressive_training'"
    echo "To attach to the session: screen -r progressive_training"
    echo "To detach from session: Ctrl+A, then D"
    exit 0
fi

echo "Running in screen session: $STY"

# 设置日志输出
exec > >(tee -a $LOG_FILE) 2>&1

echo 'Starting progressive training...'
echo "Log file: $LOG_FILE"
echo "Timestamp: $(date)"


# 设置错误处理
set -e  # 遇到错误立即退出
trap 'echo "Error occurred at line $LINENO. Check log file: $LOG_FILE"' ERR

# 添加调试信息
echo "Script started successfully"
echo "Current working directory: $(pwd)"
echo "Available conda: $(which conda)"

# 创建checkpoints目录
mkdir -p checkpoints

# ===========================================
# 基准性能评估：在原始模型上建立性能基准
# ===========================================
echo '=== BASELINE EVALUATION: Original Model Performance ==='
echo "Baseline evaluation started at: $(date)"
echo 'Creating baseline model without new parameters...'

# 创建一个不包含新参数的模型来评估原始性能
cat > create_baseline.py << 'EOF'
import torch
import sys
sys.path.append("/home/shane/code_b/unified_video_action")

# 加载原始模型配置（不包含新参数）
import hydra
from omegaconf import OmegaConf
from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy

# 使用Hydra加载配置
from hydra import initialize, compose
with initialize(config_path="unified_video_action/config"):
    cfg = compose(config_name="uva_pusht")

OmegaConf.resolve(cfg)

# 创建模型但不包含新参数
policy = UnifiedVideoActionPolicy(**cfg.model.policy, normalizer_type=cfg.task.dataset.normalizer_type, task_name=cfg.task.name)

# 加载原始checkpoint
checkpoint = torch.load("checkpoints/pusht.ckpt", map_location="cpu")
if "state_dicts" in checkpoint and "ema_model" in checkpoint["state_dicts"]:
    ema_state_dict = checkpoint["state_dicts"]["ema_model"]
    model_state_dict = {k[6:]: v for k, v in ema_state_dict.items() if k.startswith("model.")}
    policy.load_state_dict(model_state_dict, strict=False)

# 保存为基准模型（不包含新参数）
torch.save(policy.state_dict(), "checkpoints/baseline_original_model.ckpt")
EOF

/home/shane/miniforge3/condabin/conda run -n uva python create_baseline.py

echo 'Evaluating baseline model...'
echo "About to run evaluation command..."
/home/shane/miniforge3/condabin/conda run -n uva python evaluate_performance.py \
    --checkpoint='checkpoints/baseline_original_model.ckpt' \
    --phase='Baseline_Original_Model'
echo "Evaluation command completed with exit code: $?"

echo 'Baseline evaluation completed!'
echo "Baseline evaluation finished at: $(date)"

# 创建验证函数文件
cat > checkpoints/validate_params.py << 'EOF'
import torch
import sys
sys.path.append("/home/shane/code_b/unified_video_action")

def validate_unfrozen_params(policy, phase_name):
    total_params = 0
    unfrozen_params = 0
    unfrozen_names = []
    
    for name, param in policy.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            unfrozen_params += param.numel()
            unfrozen_names.append(name)
    
    return unfrozen_names
EOF

# ===========================================
# 阶段1：仅解冻local causal attention参数
# ===========================================
echo '=== PHASE 1: Training only local attention parameters ==='
echo "Phase 1 started at: $(date)"
echo 'Setting up parameter selection...'
/home/shane/miniforge3/condabin/conda run -n uva python setup_phase1.py

# 阶段1训练
echo 'Starting Phase 1 training...'
echo "Training started at: $(date)"
accelerate launch --num_processes=4 train.py \
    --config-dir=unified_video_action/config \
    --config-name=uva_pusht \
    model.policy.autoregressive_model_params.pretrained_model_path='checkpoints/pusht_phase1_local_only.ckpt' \
    model.policy.action_model_params.predict_action=True \
    training.num_epochs=25 \
    model.policy.optimizer.learning_rate=1e-4 \
    dataloader.batch_size=8 \
    logging.mode=offline \
    hydra.run.dir='checkpoints/pusht_phase1_local_attention'

echo 'Phase 1 training completed!'
echo "Phase 1 finished at: $(date)"

# 评估阶段1性能
echo 'Evaluating Phase 1 performance...'
echo "Evaluation started at: $(date)"
/home/shane/miniforge3/condabin/conda run -n uva python evaluate_performance.py \
    --checkpoint='checkpoints/pusht_phase1_local_attention/checkpoints/topk.ckpt' \
    --phase='Phase_1_Local_Attention_Only'

# 性能验证检查点
echo 'Validating Phase 1 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python validate_performance.py \
    --current-checkpoint='checkpoints/pusht_phase1_local_attention/checkpoints/topk.ckpt' \
    --baseline-checkpoint='checkpoints/baseline_original_model.ckpt' \
    --phase-name='Phase_1' \
    --threshold=0.05 \
    --output-file='checkpoints/phase1_validation.json'

if [ $? -ne 0 ]; then
    echo "❌ Phase 1 performance validation failed!"
    echo "Stopping training to prevent further degradation."
    exit 1
fi

echo "✅ Phase 1 performance validation passed!"

# ===========================================
# 阶段2：解冻部分原始模型参数
# ===========================================
echo '=== PHASE 2: Partial unfreezing for fine-tuning ==='
echo "Phase started at: $(date)"
echo 'Setting up partial parameter unfreezing...'
/home/shane/miniforge3/condabin/conda run -n uva python setup_phase2.py

# 阶段2训练
echo 'Starting Phase 2 training...'
accelerate launch --num_processes=4 train.py \
    --config-dir=unified_video_action/config \
    --config-name=uva_pusht \
    model.policy.autoregressive_model_params.pretrained_model_path='checkpoints/pusht_phase2_partial_unfrozen.ckpt' \
    model.policy.action_model_params.predict_action=True \
    training.num_epochs=20 \
    model.policy.optimizer.learning_rate=1e-4 \
    dataloader.batch_size=8 \
    logging.mode=offline \
    hydra.run.dir='checkpoints/pusht_phase2_partial_finetune'

echo 'Phase 2 completed!'

# 评估阶段2性能
echo 'Evaluating Phase 2 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python evaluate_performance.py \
    --checkpoint='checkpoints/pusht_phase2_partial_finetune/checkpoints/topk.ckpt' \
    --phase='Phase_2_Partial_Unfreezing'

# 性能验证检查点
echo 'Validating Phase 2 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python validate_performance.py \
    --current-checkpoint='checkpoints/pusht_phase2_partial_finetune/checkpoints/topk.ckpt' \
    --baseline-checkpoint='checkpoints/baseline_original_model.ckpt' \
    --phase-name='Phase_2' \
    --threshold=0.05 \
    --output-file='checkpoints/phase2_validation.json'

if [ $? -ne 0 ]; then
    echo "❌ Phase 2 performance validation failed!"
    echo "Stopping training to prevent further degradation."
    exit 1
fi

echo "✅ Phase 2 performance validation passed!"

# ===========================================
# 阶段3：解冻更多参数进行完整微调
# ===========================================
echo '=== PHASE 3: More parameters unfrozen for full fine-tuning ==='
echo "Phase started at: $(date)"
echo 'Setting up extended parameter unfreezing...'
/home/shane/miniforge3/condabin/conda run -n uva python setup_phase3.py

# 阶段3训练
echo 'Starting Phase 3 training...'
accelerate launch --num_processes=4 train.py \
    --config-dir=unified_video_action/config \
    --config-name=uva_pusht \
    model.policy.autoregressive_model_params.pretrained_model_path='checkpoints/pusht_phase3_extended_unfrozen.ckpt' \
    model.policy.action_model_params.predict_action=True \
    training.num_epochs=15 \
    model.policy.optimizer.learning_rate=3e-5 \
    dataloader.batch_size=8 \
    logging.mode=offline \
    hydra.run.dir='checkpoints/pusht_phase3_full_finetune'

echo 'Phase 3 completed!'

# 评估阶段3性能
echo 'Evaluating Phase 3 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python evaluate_performance.py \
    --checkpoint='checkpoints/pusht_phase3_full_finetune/checkpoints/topk.ckpt' \
    --phase='Phase_3_Extended_Unfreezing'

# 性能验证检查点
echo 'Validating Phase 3 performance...'
/home/shane/miniforge3/condabin/conda run -n uva python validate_performance.py \
    --current-checkpoint='checkpoints/pusht_phase3_full_finetune/checkpoints/topk.ckpt' \
    --baseline-checkpoint='checkpoints/baseline_original_model.ckpt' \
    --phase-name='Phase_3' \
    --threshold=0.05 \
    --output-file='checkpoints/phase3_validation.json'

if [ $? -ne 0 ]; then
    echo "❌ Phase 3 performance validation failed!"
    echo "Stopping training to prevent further degradation."
    exit 1
fi

echo "✅ Phase 3 performance validation passed!"

echo '=== PROGRESSIVE TRAINING COMPLETED! ==='
echo 'Final model saved in: checkpoints/pusht_phase3_full_finetune/checkpoints/topk.ckpt'
echo ''
echo '=== TRAINING SUMMARY ==='
echo 'Baseline: Original pusht.ckpt model performance'
echo 'Phase 1: Local attention parameters only (25 epochs, 1e-4 lr)'
echo 'Phase 2: + Decoder + Encoder layers 8-11 (20 epochs, 1e-4 lr)'  
echo 'Phase 3: + More encoder layers 4-11 (15 epochs, 3e-5 lr)'
echo ''
echo 'Evaluation results saved in checkpoints/ directory'

# 生成性能比较报告
echo 'Generating performance comparison report...'
/home/shane/miniforge3/condabin/conda run -n uva python compare_performance.py --checkpoints-dir='checkpoints' --output-report='progressive_training_report.txt'

echo "Progressive training started directly"
echo "All output is being logged to: $LOG_FILE"
echo "To view real-time progress: tail -f $LOG_FILE"
echo "To view the full log: cat $LOG_FILE"
