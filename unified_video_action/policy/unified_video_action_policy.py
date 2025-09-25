import torch
import os
from typing import Dict, Tuple
import torch.nn.functional as F
import random
import numpy as np

from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.common.pytorch_util import dict_apply

from unified_video_action.utils.data_utils import (
    process_data,
    extract_latent_autoregressive,
    get_trajectory,
    get_vae_latent,
    resize_image_eval,
)
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    normalize_past_action,
    unnormalize_future_action,
)
from unified_video_action.model.autoregressive import mar_con_unified as mar
from unified_video_action.vae.vaekl import AutoencoderKL
from unified_video_action.utils.language_model import (
    get_text_model,
    extract_text_features,
)


class UnifiedVideoActionPolicy(BaseImagePolicy):
    def __init__(
        self,
        vae_model_params,
        autoregressive_model_params,
        action_model_params,
        shape_meta: dict,
        n_action_steps,
        shift_action=True,
        language_emb_model=None,
        task_name=None,
        task_modes=[],
        **kwargs
    ):
        super().__init__()

        self.task_name = task_name
        self.task_modes = task_modes
        self.autoregressive_model_params = autoregressive_model_params
        self.n_action_steps = n_action_steps
        self.shift_action = shift_action
        self.language_emb_model = language_emb_model
        self.action_dim = shape_meta.action.shape[0]

        self.kwargs = kwargs
        self.normalizer_type = kwargs["normalizer_type"]
        self.selected_training_mode = kwargs["selected_training_mode"]

        self.use_history_action = kwargs["use_history_action"]
        self.use_proprioception = kwargs["use_proprioception"]

        ## =========================== load vae model ===========================
        with torch.no_grad():
            self.vae_model = AutoencoderKL(**vae_model_params)
        self.vae_model.eval()
        for param in self.vae_model.parameters():
            param.requires_grad = False

        ## =========================== load language model ===========================
        self.text_model, self.tokenizer, self.max_length = get_text_model(
            task_name, language_emb_model
        )
        if self.text_model is not None:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad = False

        ## =========================== main model ===========================
        self.model = mar.__dict__[autoregressive_model_params.model_size](
            img_size=autoregressive_model_params.img_size,
            vae_stride=autoregressive_model_params.vae_stride,
            patch_size=autoregressive_model_params.patch_size,
            vae_embed_dim=autoregressive_model_params.vae_embed_dim,
            mask_ratio_min=autoregressive_model_params.mask_ratio_min,
            label_drop_prob=autoregressive_model_params.label_drop_prob,
            attn_dropout=autoregressive_model_params.attn_dropout,
            proj_dropout=autoregressive_model_params.proj_dropout,
            diffloss_d=autoregressive_model_params.diffloss_d,
            diffloss_w=autoregressive_model_params.diffloss_w,
            diffloss_act_d=autoregressive_model_params.diffloss_act_d,
            diffloss_act_w=autoregressive_model_params.diffloss_act_w,
            num_sampling_steps=autoregressive_model_params.num_sampling_steps,
            diffusion_batch_mul=autoregressive_model_params.diffusion_batch_mul,
            grad_checkpointing=autoregressive_model_params.grad_checkpointing,
            predict_video=autoregressive_model_params.predict_video,
            act_diff_training_steps=self.autoregressive_model_params.act_diff_training_steps,
            act_diff_testing_steps=self.autoregressive_model_params.act_diff_testing_steps,
            action_model_params=action_model_params,
            use_history_action=kwargs["use_history_action"],
            action_mask_ratio=kwargs["action_mask_ratio"],
            use_proprioception=kwargs["use_proprioception"],
            predict_wrist_img=kwargs["predict_wrist_img"],
            different_history_freq=kwargs["different_history_freq"],
            predict_proprioception=kwargs["predict_proprioception"],
            task_name=self.task_name,
            language_emb_model=language_emb_model,
            shape_meta=shape_meta,
        )
        

        ## =========================== load pretrained model ===========================
        self.pretrained_model_path = autoregressive_model_params.pretrained_model_path
        if self.pretrained_model_path is not None:
            if os.path.exists(self.pretrained_model_path):
                self.load_pretrained_model()
            else:
                print('pretrained model not found: ', self.pretrained_model_path)
        
        # Apply freeze and LoRA settings after model loading
        self.apply_freeze_and_lora()
        
        self.normalizer = LinearNormalizer()

        if self.selected_training_mode is None:
            if len(self.task_modes) == 0:
                self.task_modes = [
                    "video_model",
                    "dynamic_model",
                    "policy_model",
                    "inverse_model",
                    "full_dynamic_model",
                ]
        else:
            if self.selected_training_mode == "policy_model_full_dynamics_model":
                self.task_modes = ["policy_model", "full_dynamic_model"]
            else:
                self.task_modes = [self.selected_training_mode]
        print("----------------------------------------------------------------------")
        print("task_modes", self.task_modes)
        print("----------------------------------------------------------------------")

    def load_pretrained_model(self):
        print("----------------------------------------------------------------------")
        print("Loading pretrained model: ", self.pretrained_model_path)
        print("----------------------------------------------------------------------")

        pretrained_diffusion_model_ckpt = torch.load(
            self.pretrained_model_path, map_location="cpu", weights_only=False
        )

        if "state_dicts" in pretrained_diffusion_model_ckpt:
            if "ema_model" in pretrained_diffusion_model_ckpt["state_dicts"]:
                print("load from previous ema model")
                ## load from previous checkpoint
                pretrained_diffusion_model_ckpt_ = {
                    k[6:]: v
                    for k, v in pretrained_diffusion_model_ckpt["state_dicts"][
                        "ema_model"
                    ].items()
                    if k.startswith("model.")
                }  # remove 'model.'

                model_state_dict = self.model.state_dict()
                pretrained_state_dict = {
                    k: v
                    for k, v in pretrained_diffusion_model_ckpt_.items()
                    if k in model_state_dict and model_state_dict[k].size() == v.size()
                }
                
                pretrained_state_dict_mismatch = {
                    k: v
                    for k, v in model_state_dict.items()
                    if k not in pretrained_diffusion_model_ckpt_
                    or pretrained_diffusion_model_ckpt_[k].size() != v.size()
                }
                
                print("----------------------------------------------------------------------")
                print(
                    "pretrained_state_dict_mismatch: ",
                    pretrained_state_dict_mismatch.keys(),
                )
                print("----------------------------------------------------------------------")
                
                assert len(model_state_dict) > 0
                if len(pretrained_state_dict) > 0:
                    model_state_dict.update(pretrained_state_dict)
                    print(f"Loaded {len(pretrained_state_dict)} compatible parameters from pretrained model")
                else:
                    print("Warning: No compatible parameters found in pretrained model, continuing without loading pretrained weights")
        
        # Load distilled operator weights if specified
        self.load_distilled_weights()

                missing_keys, unexpected_keys = self.model.load_state_dict(
                    model_state_dict, strict=False
                )
            else:
                raise NotImplementedError

        elif "model_ema" in pretrained_diffusion_model_ckpt:
            ## load from MAR pretrained mdoel
            pretrained_diffusion_model_ckpt_ = pretrained_diffusion_model_ckpt[
                "model_ema"
            ]

            model_state_dict = self.model.state_dict()
            pretrained_state_dict = {
                k: v
                for k, v in pretrained_diffusion_model_ckpt_.items()
                if k in model_state_dict and model_state_dict[k].size() == v.size()
            }
            assert len(model_state_dict) > 0
            assert len(pretrained_state_dict) > 0
            model_state_dict.update(pretrained_state_dict)

            missing_keys, unexpected_keys = self.model.load_state_dict(
                model_state_dict, strict=False
            )

        else:
            raise NotImplementedError

        print("---------------------------------------------------------------")
        print("Model Missing keys:", missing_keys)
        print("Model Unexpected keys:", unexpected_keys)
        print("---------------------------------------------------------------")


    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], language_goal=None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        
        obs_dict = resize_image_eval(self.task_name, obs_dict)
        B, T, C, H, W = obs_dict["image"].shape

        ## language goal
        text_latents = None
        if self.language_emb_model is not None:
            if "umi" in self.task_name:
                text_latents = language_goal
            else:
                print("predict_action language_goal: ", language_goal)
                print(self.task_name, "max_length", self.max_length)

                if self.language_emb_model == "clip":
                    text_tokens = self.tokenizer(
                        language_goal,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    text_latents = extract_text_features(
                        self.text_model,
                        text_tokens,
                        language_emb_model=self.language_emb_model,
                    )
                else:
                    text_latents = None

        ## history action
        history_nactions = None
        if self.use_history_action:
            if "past_action" in obs_dict:
                history_nactions = normalize_past_action(
                    normalizer=self.normalizer,
                    normalizer_type=self.normalizer_type,
                    actions=obs_dict["past_action"],
                )
                del obs_dict["past_action"]

        ## normalize observations
        batch = normalize_obs(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            batch={"obs": obs_dict},
        )
        obs_dict = batch["obs"]

        c, proprioception_input, _ = process_data(
            {"obs": obs_dict}, task_name=self.task_name, eval=True, **self.kwargs
        )

        if self.use_proprioception:
            if "second_image" in proprioception_input:
                second_image_z, _ = extract_latent_autoregressive(
                    self.vae_model, proprioception_input["second_image"], use_checkpointing=getattr(self, 'use_vae_checkpointing', False), chunk_size=getattr(self, 'vae_chunk_size', 4), batch_chunk_size=getattr(self, 'vae_batch_chunk_size', 8)
                )
                proprioception_input["second_image_z"] = second_image_z

        c, latent_size = extract_latent_autoregressive(self.vae_model, c.detach(), use_checkpointing=getattr(self, 'use_vae_checkpointing', False), chunk_size=getattr(self, 'vae_chunk_size', 4), batch_chunk_size=getattr(self, 'vae_batch_chunk_size', 8))

        z, act_out = self.model.sample_tokens(
            bsz=B,
            cond=c,
            text_latents=text_latents,
            num_iter=self.autoregressive_model_params.num_iter,
            cfg=self.autoregressive_model_params.cfg,
            cfg_schedule=self.autoregressive_model_params.cfg_schedule,
            temperature=self.autoregressive_model_params.temperature,
            history_nactions=history_nactions,
            proprioception_input=proprioception_input,
            task_mode="policy_model",
            vae_model=self.vae_model,
        )

        # unnormalize prediction
        Da = self.action_dim

        naction_pred = act_out[..., :Da]

        ## unnormalize action
        action_pred = unnormalize_future_action(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            actions=naction_pred,
        )

        action = action_pred[:, : self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def add_weight_decay(self, model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)  # no weight decay on bias, norm and diffloss
            else:
                decay.append(param)

        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:

        decay = []
        no_decay = []
        local_decay = []
        local_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "local_causal_blocks" in name or 'feature_fusion' in name:
                if len(param.shape) == 1 or name.endswith(".bias"):
                    local_no_decay.append(param)
                else:
                    local_decay.append(param)
            else:
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
                    
        return torch.optim.AdamW([
            #原来的参数 - 小学习率
            {"params": no_decay, "weight_decay": 0.0, "lr": learning_rate * 0.1},
            {"params": decay, "weight_decay": weight_decay, "lr": learning_rate * 0.1},
            #新的参数 - 正常的学习率
            {"params": local_no_decay, "weight_decay": 0.0, "lr": learning_rate},
            {"params": local_decay, "weight_decay": weight_decay, "lr": learning_rate},
        ],betas=betas)

    def compute_loss(self, batch, **kwargs):
        B, T, C, H, W = batch["obs"]["image"].size()

        text_latents = None
        if self.language_emb_model == "clip":
            if "language" in batch["obs"]:
                language_goal = batch["obs"]["language"]
                del batch["obs"]["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    self.text_model,
                    text_tokens,
                    language_emb_model=self.language_emb_model,
                )
            elif "language_latents" in batch:
                text_latents = batch["language_latents"]
            else:
                raise NotImplementedError

        nactions = normalize_action(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            actions=batch["action"],
        )
        batch = normalize_obs(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            batch=batch,
        )

        if self.use_history_action:
            batch = dict_apply(batch, lambda x: x[:, 1:])

        x, proprioception_input, _ = process_data(
            batch, task_name=self.task_name, **self.kwargs
        )
        x, z, c, _, proprioception_input = get_vae_latent(
            x, self.vae_model, eval=False, proprioception_input=proprioception_input
        )
        history_trajectory, trajectory = get_trajectory(
            nactions, T, self.shift_action, use_history_action=self.use_history_action
        )

        selected_mode = random.choice(self.task_modes)

        loss, video_loss, act_loss = self.model(
            z,
            c,
            history_trajectory,
            trajectory,
            text_latents,
            task_mode=selected_mode,
            proprioception_input=proprioception_input,
        )

        ## not recommended, fix the problem in DDM unused parameters
        for param in self.model.parameters():
            if param.grad is None:  # Likely unused in loss computation
                loss += 0 * param.sum()

        return loss, (video_loss, act_loss)

    def forward(self, batch, **kwargs):
        return self.compute_loss(batch, **kwargs)
    
    def load_distilled_weights(self):
        """Load distilled operator weights for cross-attention head"""
        # Check if head_init_paths is specified in action_model_params
        if hasattr(self.autoregressive_model_params, 'action_model_params') and \
           hasattr(self.autoregressive_model_params.action_model_params, 'head_init_paths'):
            
            head_init_paths = self.autoregressive_model_params.action_model_params.head_init_paths
            
            if head_init_paths and any(head_init_paths.values()):
                print("----------------------------------------------------------------------")
                print("Loading distilled operator weights for cross-attention head")
                print("----------------------------------------------------------------------")
                
                # Load cross-attention weights
                if head_init_paths.get('cross_attn') and os.path.exists(head_init_paths['cross_attn']):
                    print(f"Loading cross-attention weights from: {head_init_paths['cross_attn']}")
                    cross_attn_weights = torch.load(head_init_paths['cross_attn'], map_location='cpu')
                    # Load into the diffusion action head
                    if hasattr(self.model, 'diffactloss') and hasattr(self.model.diffactloss, 'net'):
                        net = self.model.diffactloss.net
                        if hasattr(net, 'cross_attn_blocks'):
                            # Load into first cross-attention block
                            net.cross_attn_blocks[0].load_state_dict(cross_attn_weights, strict=False)
                            print("Cross-attention weights loaded successfully")
                
                # Load self-attention weights
                if head_init_paths.get('self_attn') and os.path.exists(head_init_paths['self_attn']):
                    print(f"Loading self-attention weights from: {head_init_paths['self_attn']}")
                    self_attn_weights = torch.load(head_init_paths['self_attn'], map_location='cpu')
                    # Load into the diffusion action head
                    if hasattr(self.model, 'diffactloss') and hasattr(self.model.diffactloss, 'net'):
                        net = self.model.diffactloss.net
                        if hasattr(net, 'cross_attn_blocks'):
                            # Load into first cross-attention block (reusing the same structure)
                            net.cross_attn_blocks[0].load_state_dict(self_attn_weights, strict=False)
                            print("Self-attention weights loaded successfully")
                
                # Load MLP weights
                if head_init_paths.get('mlp') and os.path.exists(head_init_paths['mlp']):
                    print(f"Loading MLP weights from: {head_init_paths['mlp']}")
                    mlp_weights = torch.load(head_init_paths['mlp'], map_location='cpu')
                    # Load into the diffusion action head
                    if hasattr(self.model, 'diffactloss') and hasattr(self.model.diffactloss, 'net'):
                        net = self.model.diffactloss.net
                        if hasattr(net, 'cross_attn_blocks'):
                            # Load into first cross-attention block's MLP
                            net.cross_attn_blocks[0].mlp.load_state_dict(mlp_weights, strict=False)
                            print("MLP weights loaded successfully")
                
                print("----------------------------------------------------------------------")
                print("Distilled weights loading completed")
                print("----------------------------------------------------------------------")
            else:
                print("No distilled weights specified, using random initialization")
    
    def apply_freeze_and_lora(self):
        """Apply parameter freezing and LoRA if specified"""
        # Check if finetune config is available
        if hasattr(self.autoregressive_model_params, 'action_model_params') and \
           hasattr(self.autoregressive_model_params.action_model_params, 'finetune'):
            
            finetune_config = self.autoregressive_model_params.action_model_params.finetune
            
            if finetune_config.get('freeze_encoder', False):
                print("----------------------------------------------------------------------")
                print("Freezing encoder parameters")
                print("----------------------------------------------------------------------")
                
                # Freeze encoder blocks
                for param in self.model.encoder_blocks.parameters():
                    param.requires_grad = False
                
                # Freeze local causal encoder blocks
                for param in self.model.local_causal_encoder_blocks.parameters():
                    param.requires_grad = False
                
                # Freeze encoder norm
                for param in self.model.encoder_norm.parameters():
                    param.requires_grad = False
                
                # Freeze local causal encoder norm
                for param in self.model.local_causal_encoder_norm.parameters():
                    param.requires_grad = False
                
                # Freeze feature fusion
                for param in self.model.feature_fusion.parameters():
                    param.requires_grad = False
                
                print("Encoder parameters frozen")
            
            if finetune_config.get('use_lora', False):
                print("----------------------------------------------------------------------")
                print("LoRA support not yet implemented")
                print("----------------------------------------------------------------------")
                # TODO: Implement LoRA support
                pass
