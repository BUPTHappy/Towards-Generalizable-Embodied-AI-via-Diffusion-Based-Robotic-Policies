#!/usr/bin/env python3
"""
Stage 1: Cache teacher activations for distillation training
缓存教师模型的激活用于蒸馏训练
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

# Import your existing modules
from unified_video_action.dataset import get_dataset
from unified_video_action.model.autoregressive.mar_con_unified import MAR
from unified_video_action.model.autoregressive.diffusion_action_loss import DiffActLoss
from unified_video_action.model.autoregressive.diffusion_loss import SimpleMLPAdaLN
from unified_video_action.model.autoregressive.cross_attention_diffusion import CrossAttentionAdaLN


class TeacherActivationCache:
    """缓存教师模型的激活"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook to capture activations
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册钩子函数来捕获激活"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu()
            return hook
        
        # Hook the diffusion action head components
        if hasattr(self.model, 'diffactloss') and hasattr(self.model.diffactloss, 'net'):
            net = self.model.diffactloss.net
            
            # Hook SimpleMLPAdaLN components
            if isinstance(net, SimpleMLPAdaLN):
                # Hook input projection
                self.hooks.append(net.input_proj.register_forward_hook(get_activation('input_proj')))
                
                # Hook each residual block
                for i, block in enumerate(net.res_blocks):
                    self.hooks.append(block.register_forward_hook(get_activation(f'res_block_{i}')))
                
                # Hook final layer
                self.hooks.append(net.final_layer.register_forward_hook(get_activation('final_layer')))
                
            # Hook CrossAttentionAdaLN components (if used)
            elif isinstance(net, CrossAttentionAdaLN):
                # Hook input projection
                self.hooks.append(net.input_proj.register_forward_hook(get_activation('input_proj')))
                
                # Hook video projection
                self.hooks.append(net.video_proj.register_forward_hook(get_activation('video_proj')))
                
                # Hook each cross attention block
                for i, block in enumerate(net.cross_attn_blocks):
                    self.hooks.append(block.register_forward_hook(get_activation(f'cross_attn_block_{i}')))
                
                # Hook final layer
                self.hooks.append(net.final_layer.register_forward_hook(get_activation('final_layer')))
    
    def remove_hooks(self):
        """移除钩子函数"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def cache_activations(self, dataloader, num_samples=8000):
        """缓存激活数据"""
        cached_data = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching activations")):
                if len(cached_data) >= num_samples:
                    break
                
                # Clear activations before each batch
                self.activations.clear()
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                try:
                    # Try different forward call patterns
                    if hasattr(self.model, 'forward'):
                        result = self.model(
                            imgs=batch['image'],
                            cond=batch['cond'],
                            nactions=batch['action'],
                            task_mode="policy_model"
                        )
                        
                        # Handle different return formats
                        if isinstance(result, tuple):
                            if len(result) == 3:
                                loss, video_loss, act_loss = result
                            elif len(result) == 2:
                                loss, video_loss = result
                                act_loss = torch.tensor(0.0)
                            else:
                                loss = result[0]
                                video_loss = torch.tensor(0.0)
                                act_loss = torch.tensor(0.0)
                        else:
                            loss = result
                            video_loss = torch.tensor(0.0)
                            act_loss = torch.tensor(0.0)
                    else:
                        raise AttributeError("Model has no forward method")
                    
                    # Only store if we have activations
                    if self.activations:
                        batch_data = {
                            'batch_idx': batch_idx,
                            'activations': self.activations.copy(),
                            'input_shape': batch['image'].shape,
                            'cond_shape': batch['cond'].shape,
                            'action_shape': batch['action'].shape,
                        }
                        
                        cached_data.append(batch_data)
                    else:
                        print(f"Warning: No activations captured for batch {batch_idx}")
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Clear activations on error to prevent accumulation
                    self.activations.clear()
                    continue
        
        return cached_data


def main():
    parser = argparse.ArgumentParser(description='Cache teacher activations for distillation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for cached activations')
    parser.add_argument('--num_samples', type=int, default=8000,
                       help='Number of samples to cache (default: 8000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for caching (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load teacher model
    print(f"Loading teacher model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract model config from checkpoint - try multiple possible keys
    model_config = None
    for key in ['model_config', 'config', 'hyper_parameters', 'hparams']:
        if key in checkpoint:
            model_config = checkpoint[key]
            print(f"Found model config in checkpoint key: {key}")
            break
    
    if model_config is None:
        print("Warning: No model config found in checkpoint, using defaults")
        model_config = {
            'encoder_embed_dim': 1024,
            'decoder_embed_dim': 1024,
            'predict_action': True,
            'action_model_params': {
                'predict_action': True,
                'act_model_type': 'conv_fc',  # Use original MLP head
                'num_attention_heads': 8
            }
        }
    
    # Create model with safe parameter handling
    try:
        model = MAR(**model_config)
    except Exception as e:
        print(f"Error creating model with config: {e}")
        print("Falling back to default config...")
        model_config = {
            'encoder_embed_dim': 1024,
            'decoder_embed_dim': 1024,
            'predict_action': True,
            'action_model_params': {
                'predict_action': True,
                'act_model_type': 'conv_fc',
                'num_attention_heads': 8
            }
        }
        model = MAR(**model_config)
    
    # Safe state dict loading
    model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    if not model_state_dict:
        print("Warning: No model state dict found in checkpoint")
        model_state_dict = {}
    
    # Load with strict=False and handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys in checkpoint: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:  # Only print if not too many
            print(f"Missing keys: {missing_keys}")
    
    if unexpected_keys:
        print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:  # Only print if not too many
            print(f"Unexpected keys: {unexpected_keys}")
    
    model = model.to(args.device)
    model.eval()
    
    # Create activation cache
    cache = TeacherActivationCache(model, args.device)
    cache.register_hooks()
    
    # Load dataset with proper error handling
    print(f"Loading dataset from {args.dataset}")
    try:
        # Try different dataset loading approaches
        if hasattr(get_dataset, '__call__'):
            # If get_dataset is a function, try calling it
            try:
                dataset = get_dataset(args.dataset)
            except Exception as e:
                print(f"Error calling get_dataset({args.dataset}): {e}")
                # Fallback: try with additional parameters
                try:
                    dataset = get_dataset(args.dataset, split='train')
                except Exception as e2:
                    print(f"Error with split parameter: {e2}")
                    raise e
        else:
            # If get_dataset is a class, instantiate it
            dataset = get_dataset(args.dataset)
        
        # Verify dataset is properly loaded
        if not hasattr(dataset, '__len__'):
            raise AttributeError("Dataset does not have __len__ method")
        if not hasattr(dataset, '__getitem__'):
            raise AttributeError("Dataset does not have __getitem__ method")
        
        print(f"Dataset loaded successfully: {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your dataset path and ensure get_dataset function is properly imported")
        raise
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Cache activations with proper error handling
    print(f"Caching activations for {args.num_samples} samples...")
    try:
        cached_data = cache.cache_activations(dataloader, args.num_samples)
        
        if not cached_data:
            raise RuntimeError("No activations were cached. Check model hooks and forward pass.")
        
        # Save cached data with memory optimization
        output_file = os.path.join(args.out, 'teacher_activations.pkl')
        
        # Process activations to reduce memory usage
        processed_data = []
        for batch_data in cached_data:
            processed_batch = {
                'batch_idx': batch_data['batch_idx'],
                'input_shape': batch_data['input_shape'],
                'cond_shape': batch_data['cond_shape'],
                'action_shape': batch_data['action_shape'],
            }
            
            # Process activations to reduce memory
            processed_activations = {}
            for key, tensor in batch_data['activations'].items():
                # Convert to float16 to save memory
                if tensor.dtype == torch.float32:
                    tensor = tensor.half()
                # Move to CPU if on GPU
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                processed_activations[key] = tensor
            
            processed_batch['activations'] = processed_activations
            processed_data.append(processed_batch)
        
        # Save with compression
        print(f"Saving cached data to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Cached {len(processed_data)} samples to {output_file}")
        
        # Calculate and print memory usage
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        # Save metadata
        metadata = {
            'num_samples': len(cached_data),
            'model_config': model_config,
            'checkpoint_path': args.checkpoint,
            'dataset_path': args.dataset,
            'batch_size': args.batch_size,
        }
        
        metadata_file = os.path.join(args.out, 'metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Metadata saved to {metadata_file}")
        
    except Exception as e:
        print(f"Error during caching: {e}")
        raise
    finally:
        # Always clean up resources
        try:
            cache.remove_hooks()
            print("Hooks removed successfully")
        except Exception as e:
            print(f"Warning: Error removing hooks: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")


if __name__ == '__main__':
    main()
