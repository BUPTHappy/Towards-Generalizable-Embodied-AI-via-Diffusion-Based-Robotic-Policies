#!/bin/bash

echo "Starting training in screen session..."

# 禁用WandB
export WANDB_MODE=disabled

# 创建screen会话并运行训练
screen -dmS local_attention_training bash -c "
accelerate launch --num_processes=4 train.py \
    --config-dir=. \
    --config-name=uva_pusht.yaml \
    model.policy.autoregressive_model_params.pretrained_model_path='checkpoints/pusht.ckpt' \
    model.policy.action_model_params.predict_action=True \
    training.num_epochs=5 \
    model.policy.optimizer.learning_rate=1e-5 \
    dataloader.batch_size=8 \
    logging.mode=offline \
    hydra.run.dir='checkpoints/pusht_local_attention_quick'
"

echo "Training started in screen session: local_attention_training"
echo "To attach to session: screen -r local_attention_training"
echo "To detach from session: Ctrl+A, then D"
echo "To list sessions: screen -ls"