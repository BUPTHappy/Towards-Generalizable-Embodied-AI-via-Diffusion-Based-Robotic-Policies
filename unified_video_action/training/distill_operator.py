#!/usr/bin/env python3
"""
Stage 1: Distill individual operators (cross-attention, self-attention, MLP)
蒸馏单个算子（交叉注意力、自注意力、MLP）
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

# Import your existing modules
from unified_video_action.model.autoregressive.cross_attention_diffusion import CrossAttentionAdaLN, CrossAttentionBlock
from unified_video_action.model.autoregressive.diffusion_loss import SimpleMLPAdaLN


class OperatorDistiller:
    """单个算子的蒸馏训练器"""
    
    def __init__(self, target_layer, device='cuda'):
        self.target_layer = target_layer
        self.device = device
        
    def create_student_operator(self, teacher_activations, model_channels=1024, num_heads=8):
        """根据教师激活创建学生算子"""
        
        if 'cross_attn' in self.target_layer:
            # Create cross-attention student
            student = CrossAttentionBlock(
                dim=model_channels,
                num_heads=num_heads,
                dropout=0.1
            )
            
        elif 'self_attn' in self.target_layer:
            # Create self-attention student (reuse CrossAttentionBlock but only use self-attention)
            student = CrossAttentionBlock(
                dim=model_channels,
                num_heads=num_heads,
                dropout=0.1
            )
            
        elif 'mlp' in self.target_layer:
            # Create MLP student
            student = nn.Sequential(
                nn.Linear(model_channels, model_channels * 4, bias=True),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(model_channels * 4, model_channels, bias=True),
                nn.Dropout(0.1),
            )
            
        else:
            raise ValueError(f"Unknown target layer: {self.target_layer}")
        
        return student.to(self.device)
    
    def prepare_training_data(self, cached_activations, target_layer):
        """准备训练数据"""
        inputs = []
        targets = []
        
        print(f"Processing {len(cached_activations)} batches for layer: {target_layer}")
        
        for batch_idx, batch_data in enumerate(cached_activations):
            activations = batch_data['activations']
            
            # Find the target activation
            if target_layer in activations:
                target_activation = activations[target_layer]
                
                # Convert from half precision if needed
                if target_activation.dtype == torch.float16:
                    target_activation = target_activation.float()
                
                # Prepare input based on layer type
                if 'cross_attn' in target_layer:
                    # For cross-attention, we need both action and video features
                    if 'input_proj' in activations:
                        input_data = activations['input_proj']
                        if input_data.dtype == torch.float16:
                            input_data = input_data.float()
                        inputs.append(input_data)
                        targets.append(target_activation)
                        
                elif 'self_attn' in target_layer:
                    # For self-attention, input is the same as target
                    if 'input_proj' in activations:
                        input_data = activations['input_proj']
                        if input_data.dtype == torch.float16:
                            input_data = input_data.float()
                        inputs.append(input_data)
                        targets.append(target_activation)
                        
                elif 'mlp' in target_layer:
                    # For MLP, input is the same as target
                    if 'input_proj' in activations:
                        input_data = activations['input_proj']
                        if input_data.dtype == torch.float16:
                            input_data = input_data.float()
                        inputs.append(input_data)
                        targets.append(target_activation)
            else:
                print(f"Warning: Layer {target_layer} not found in batch {batch_idx}")
        
        if not inputs:
            raise ValueError(f"No training data found for layer: {target_layer}")
        
        print(f"Found {len(inputs)} valid samples")
        
        # Concatenate all inputs and targets with proper error handling
        try:
            inputs = torch.cat(inputs, dim=0)
            targets = torch.cat(targets, dim=0)
        except Exception as e:
            print(f"Error concatenating tensors: {e}")
            print(f"Input shapes: {[inp.shape for inp in inputs[:5]]}")
            print(f"Target shapes: {[tgt.shape for tgt in targets[:5]]}")
            raise
        
        print(f"Final data shapes: inputs={inputs.shape}, targets={targets.shape}")
        return inputs, targets
    
    def train_student(self, student, inputs, targets, epochs=200, lr=1e-4, batch_size=64):
        """训练学生算子"""
        
        # Create dataset and dataloader
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Choose loss function based on layer type
        if 'attn' in self.target_layer:
            criterion = nn.L1Loss()  # L1 for attention layers
        else:
            criterion = nn.MSELoss()  # L2 for MLP layers
        
        # Training loop with proper error handling
        student.train()
        best_loss = float('inf')
        
        try:
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for batch_inputs, batch_targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                    try:
                        batch_inputs = batch_inputs.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        optimizer.zero_grad()
                        
                        # Forward pass with proper error handling
                        if 'cross_attn' in self.target_layer:
                            # For cross-attention, we need video features as well
                            output = student(batch_inputs, batch_inputs, batch_inputs.mean(dim=1))
                        elif 'self_attn' in self.target_layer:
                            # For self-attention, query=key=value=input
                            output = student(batch_inputs, batch_inputs, batch_inputs.mean(dim=1))
                        else:
                            # For MLP
                            output = student(batch_inputs)
                        
                        # Check for NaN or inf values
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            print(f"Warning: NaN or inf detected in output at epoch {epoch+1}")
                            continue
                        
                        loss = criterion(output, batch_targets)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN or inf loss at epoch {epoch+1}")
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Check for gradient issues
                        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"Warning: NaN or inf gradients at epoch {epoch+1}")
                            continue
                        
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                        
                    except Exception as e:
                        print(f"Error in batch at epoch {epoch+1}: {e}")
                        continue
                
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
                    
                    # Track best loss
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                else:
                    print(f"Epoch {epoch+1}/{epochs}: No valid batches processed")
                    
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            raise
        
        print(f"Training completed. Best loss: {best_loss:.6f}")
        return student


def main():
    parser = argparse.ArgumentParser(description='Distill individual operators')
    parser.add_argument('--cached_dir', type=str, required=True,
                       help='Directory containing cached activations')
    parser.add_argument('--target_layer', type=str, required=True,
                       help='Target layer to distill (e.g., cross_attn, self_attn, mlp)')
    parser.add_argument('--loss', type=str, default='l1',
                       choices=['l1', 'l2', 'huber'],
                       help='Loss function to use')
    parser.add_argument('--out', type=str, required=True,
                       help='Output path for distilled weights')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load cached activations with proper error handling
    cached_file = os.path.join(args.cached_dir, 'teacher_activations.pkl')
    metadata_file = os.path.join(args.cached_dir, 'metadata.pkl')
    
    if not os.path.exists(cached_file):
        raise FileNotFoundError(f"Cached activations not found: {cached_file}")
    
    print(f"Loading cached activations from {cached_file}")
    try:
        with open(cached_file, 'rb') as f:
            cached_activations = pickle.load(f)
    except Exception as e:
        print(f"Error loading cached activations: {e}")
        raise
    
    # Load metadata
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = {}
    
    print(f"Loaded {len(cached_activations)} cached samples")
    
    # Create distiller
    distiller = OperatorDistiller(args.target_layer, args.device)
    
    # Prepare training data
    print(f"Preparing training data for {args.target_layer}")
    inputs, targets = distiller.prepare_training_data(cached_activations, args.target_layer)
    
    print(f"Training data shape: inputs={inputs.shape}, targets={targets.shape}")
    
    # Create student operator
    model_channels = metadata.get('model_config', {}).get('decoder_embed_dim', 1024)
    num_heads = metadata.get('model_config', {}).get('action_model_params', {}).get('num_attention_heads', 8)
    
    student = distiller.create_student_operator(
        cached_activations, 
        model_channels=model_channels,
        num_heads=num_heads
    )
    
    print(f"Created student operator: {student}")
    
    # Train student with proper error handling
    print(f"Training student operator for {args.epochs} epochs...")
    try:
        trained_student = distiller.train_student(
            student, inputs, targets,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
        
        # Save trained weights
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(trained_student.state_dict(), args.out)
        
        print(f"Distilled weights saved to {args.out}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")


if __name__ == '__main__':
    main()
