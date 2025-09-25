# 两阶段训练方案：Diffusion Action Head Grafting

这个方案将你的diffusion action head从MLP架构安全地迁移到cross-attention + self-attention + MLP结构，通过两阶段训练确保性能不下降。

## 方案概述

### Stage 1 — 激活蒸馏（逐算子初始化）
- **目标**：为每个新算子单独训练一个"学生算子"使其输出逼近旧MLP-head在相同输入下的激活
- **步骤**：
  1. 缓存教师模型激活（8k-16k样本）
  2. 并行蒸馏cross-attention、self-attention、MLP算子
  3. 保存每个算子的初始化权重

### Stage 2 — 端到端轻量微调
- **目标**：将蒸馏好的算子装回模型并做小规模端到端微调
- **策略**：冻结encoder，仅微调action head，使用10%数据，50k步数

## 文件结构

```
unified_video_action/
├── tools/
│   └── cache_activation.py          # Stage 1: 缓存教师激活
├── training/
│   └── distill_operator.py         # Stage 1: 算子蒸馏训练
├── config/
│   └── train_diffusion_graft_workspace.yaml  # Stage 2: 配置文件
├── policy/
│   └── unified_video_action_policy.py       # 修改：支持权重加载和冻结
└── model/autoregressive/
    ├── diffusion_action_loss.py             # 修改：支持cross_attention模式
    └── cross_attention_diffusion.py         # 新的cross-attention实现
```

## 使用方法

### 完整两阶段训练

```bash
# 运行完整的两阶段训练
bash scripts/training/train_graft_head.sh
```

### 仅Stage 2测试（假设Stage 1已完成）

```bash
# 快速测试Stage 2
bash scripts/training/quick_test_graft_head.sh
```

### 手动执行

#### Stage 1: 缓存激活
```bash
CUDA_VISIBLE_DEVICES=0 python unified_video_action/tools/cache_activation.py \
    --checkpoint checkpoints/pusht.ckpt \
    --dataset data/pusht/ \
    --out cached_acts/ \
    --num_samples 8000
```

#### Stage 1: 蒸馏算子
```bash
# 蒸馏cross-attention
CUDA_VISIBLE_DEVICES=0 python unified_video_action/training/distill_operator.py \
    --cached_dir cached_acts/ \
    --target_layer cross_attn \
    --loss l1 \
    --out init_weights/cross_attn.pt

# 蒸馏self-attention
CUDA_VISIBLE_DEVICES=0 python unified_video_action/training/distill_operator.py \
    --cached_dir cached_acts/ \
    --target_layer self_attn \
    --loss l1 \
    --out init_weights/self_attn.pt

# 蒸馏MLP
CUDA_VISIBLE_DEVICES=0 python unified_video_action/training/distill_operator.py \
    --cached_dir cached_acts/ \
    --target_layer mlp \
    --loss l2 \
    --out init_weights/mlp.pt
```

#### Stage 2: 端到端微调
```bash
accelerate launch --num_processes=1 train.py \
    --config-name=train_diffusion_graft_workspace \
    model.policy.action_model_params.head_init_paths.cross_attn=init_weights/cross_attn.pt \
    model.policy.action_model_params.head_init_paths.self_attn=init_weights/self_attn.pt \
    model.policy.action_model_params.head_init_paths.mlp=init_weights/mlp.pt \
    model.policy.finetune.freeze_encoder=true \
    training.steps=50000 \
    training.data_fraction=0.10 \
    model.policy.optimizer.learning_rate=1e-4 \
    hydra.run.dir="checkpoints/uva_graft_head"
```

## 配置说明

### 关键配置参数

- `model.policy.action_model_params.act_model_type: cross_attention` - 使用新的cross-attention头
- `model.policy.action_model_params.head_init_paths` - 蒸馏权重的路径
- `model.policy.finetune.freeze_encoder: true` - 冻结encoder参数
- `training.data_fraction: 0.10` - 使用10%数据微调
- `training.steps: 50000` - 微调步数

### 超参数建议

- **Stage 1**: epochs=200, lr=1e-4, batch_size=64
- **Stage 2**: lr=1e-4, warmup=1k steps, batch_size=256
- **数据量**: 起始10%，效果不佳可增至20%
- **冻结策略**: 先只训练head 5k-10k steps

## 监控指标

训练期间关注以下指标：
- `video_fvd` - 视频生成质量
- `action accuracy/mAP` - 动作预测准确性
- `test_mean_score` - 整体任务性能

## 故障排除

### 常见问题

1. **训练后性能下降**
   - 检查Stage 1的回归loss
   - 增加Stage 1的数据量或训练步数
   - 调整学习率

2. **内存不足**
   - 减少batch_size
   - 启用gradient_checkpointing
   - 使用LoRA（设置`use_lora: true`）

3. **权重加载失败**
   - 检查权重文件路径
   - 确认模型结构匹配
   - 使用`strict=False`加载

### 调试建议

- 先用小数据集测试（data_fraction=0.01）
- 监控Stage 1的蒸馏loss收敛情况
- 逐步放开冻结参数进行调试

## 预期结果

- **Stage 1**: 各算子回归loss应收敛到较低值（<0.01）
- **Stage 2**: 微调后性能应接近或超过原始MLP头
- **整体**: 新架构应保持原有任务性能，同时具备更好的表示能力
