from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats #用于生成随机掩码比例
import math
from einops import rearrange #用于张量维度重排
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint #用于节省显存的梯度检查点技术

from timm.models.vision_transformer import Block
from unified_video_action.model.autoregressive.diffusion_loss import DiffLoss
from unified_video_action.model.autoregressive.diffusion_action_loss import DiffActLoss

# from unified_video_action.model.autoregressive.local_causal_attention import LocalCausalAttentionBlock

class LocalCausalTransformerBlock(nn.Module):
    """Transformer block with local causal attention window mechanism"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, 
                 proj_drop=0.0, attn_drop=0.0, window_size=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LocalCausalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                       attn_drop=attn_drop, proj_drop=proj_drop, window_size=window_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LocalCausalAttention(nn.Module):
    """Multi-head attention with local causal window mechanism"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def create_local_causal_mask(self, seq_len):
        """Create local causal mask with window size"""
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            # Each position can only attend to current and previous window_size-1 positions
            start_idx = max(0, i - self.window_size + 1)
            mask[i, start_idx:i+1] = 1
            
        return mask == 0  # True for positions that should be masked

    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Scale attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply local causal mask
        local_causal_mask = self.create_local_causal_mask(N).to(x.device)
        attn = attn.masked_fill(local_causal_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#根据预设的顺序创建掩码
def mask_by_order(mask_len, order, bsz, seq_len, device):
    masking = torch.zeros(bsz, seq_len).to(device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).to(device),
    ).bool()
    return masking


class MAR(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        diffloss_d=3,
        diffloss_w=1024,
        diffloss_act_d=3,
        diffloss_act_w=1024,
        num_sampling_steps="100",
        diffusion_batch_mul=4,
        grad_checkpointing=False,
        predict_video=True,
        act_diff_training_steps=1000,
        act_diff_testing_steps="100",
        action_model_params={},
        **kwargs
    ):
        super().__init__()

        self.task_name = kwargs["task_name"] # 任务名称
        self.different_history_freq = kwargs["different_history_freq"]
        self.use_history_action = kwargs["use_history_action"]
        self.action_mask_ratio = kwargs["action_mask_ratio"] # 动作掩码比例
        self.use_proprioception = kwargs["use_proprioception"] # 是否使用感知
        self.predict_wrist_img = kwargs["predict_wrist_img"] # 是否预测手腕图像
        self.predict_proprioception = kwargs["predict_proprioception"] # 是否预测感知
        self.n_frames = 4 # 固定处理4帧视频

        # ========= VAE and patchify specifics =========
            # 输入图像 256x256 → VAE编码后 16x16 → 每个位置是一个token
            # 总共有 16×16=256 个空间位置的token
            # 4帧视频 × 256个空间token = 1024个时空token
        
        self.img_size = img_size # 图像大小 256x256
        self.vae_stride = vae_stride # VAE下采样倍数 16
        self.patch_size = patch_size # 补丁大小
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size # 16x16=256个空间位置
        self.seq_len = self.seq_h * self.seq_w # 256个空间位置
        self.token_embed_dim = vae_embed_dim * patch_size**2  
        self.vae_embed_dim = vae_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.label_drop_prob = label_drop_prob
        
        # ========= Masked MAE =========
        # variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
            # 掩码比例生成器，左半截断高斯分布，中心在100%，标准差为0.25： 大部分时候会掩盖70%-100%的token
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # ========= Projection =========
        # conditional frames 条件帧投影（历史观察帧）
        self.z_proj_cond = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)

        # video frames 目标帧投影（要预测的视频帧）
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)

        # wrist video frames 手腕相机投影（如果预测手腕图像）
        if self.predict_wrist_img:
            self.z_proj_wrist = nn.Linear(
                self.token_embed_dim, encoder_embed_dim, bias=True
            )

        # action 动作
        self.predict_action = action_model_params["predict_action"]
        act_dim = kwargs["shape_meta"]["action"]["shape"][0]

        self.action_proj_cond = nn.Linear(act_dim, encoder_embed_dim, bias=True)
        self.buffer_size_action = 64

        # ========= Fake Latent =========
         
        self.fake_latent_x = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.fake_action_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        if self.predict_wrist_img:
            self.fake_latent_wrist_x = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        if self.use_history_action:
            self.fake_latent_history_action = nn.Parameter(
                torch.zeros(1, encoder_embed_dim)
            )

        # ========= History Action =========
        if self.use_history_action:
            self.history_action_proj_cond = nn.Linear(
                act_dim, encoder_embed_dim, bias=True
            )

        # ========= Proprioception =========
        if self.use_proprioception:
            self.buffer_size_properception = 64
            if self.different_history_freq:
                self.buffer_size_properception = 64 * 4

            if self.task_name == "umi":
                self.proprioception_proj_cond = nn.Linear(
                    16, encoder_embed_dim, bias=True
                )
            elif "pusht" in self.task_name:
                self.proprioception_proj_cond = nn.Linear(
                    2, encoder_embed_dim, bias=True
                )
            else:
                self.proprioception_proj_cond = nn.Linear(
                    9, encoder_embed_dim, bias=True
                )

            self.proprioception_image_proj_cond = nn.Linear(
                self.token_embed_dim, encoder_embed_dim, bias=True
            )

        # ========= Language Embedding =========
        self.language_emb_model = kwargs["language_emb_model"]
        self.language_emb_model_type = 1

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
                self.text_proj_cond = nn.Linear(
                    512, encoder_embed_dim, bias=True
                )  # clip text embedding is 512
                self.buffer_size_text = 64
                self.text_pos_embed = nn.Parameter(
                    torch.zeros(1, self.buffer_size_text, encoder_embed_dim)
                )

        # ========= Projection =========
        if self.predict_wrist_img:
            proj_cond_x_dim_num = 4
            if self.use_proprioception:
                proj_cond_x_dim_num += 2
            if self.use_history_action:
                proj_cond_x_dim_num += 1
        else:
            proj_cond_x_dim_num = 3
            if self.use_proprioception:
                if (
                    self.task_name == "umi"
                    or "block_push" in self.task_name
                    or "pusht" in self.task_name
                ):
                    proj_cond_x_dim_num += 1
                else:
                    proj_cond_x_dim_num += 2
            if self.use_history_action:
                proj_cond_x_dim_num += 1

        self.proj_cond_x_layer = nn.Linear(
            proj_cond_x_dim_num * encoder_embed_dim, encoder_embed_dim, bias=True
        )

        # ========= Temporal and Spatial Position Embedding =========
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, encoder_embed_dim)
        )  # Temporal position embedding, 4 frames
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, encoder_embed_dim)
        )  # Spatial position embedding

        # ========= Normalization =========
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # ========= Encoder Blocks =========
        # Original transformer blocks
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # ========= Local Causal Encoder Blocks (copy of original with window mechanism) =========
        self.local_causal_encoder_blocks = nn.ModuleList(
            [
                LocalCausalTransformerBlock(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                    window_size=15,  # 增加窗口大小，从3改为8，提供更大的感受野
                )
                for _ in range(encoder_depth)
            ]
        )
        self.local_causal_encoder_norm = norm_layer(encoder_embed_dim)
        
        # ========= Feature Fusion =========
        # 极简的特征融合：只用一个线性层
        self.feature_fusion = nn.Linear(encoder_embed_dim * 2, encoder_embed_dim)
        
        # 使用简单的λ参数控制特征融合权重，避免训练门控网络
        self.lambda_local = 0.1  # 使用很小的权重，避免破坏原始特征

        # ========= Decoder =========
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # ========= Decoder Position Embedding =========
        self.decoder_temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, decoder_embed_dim)
        )
        self.decoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        # ========= Decoder Text Position Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                self.decoder_text_pos_embed = nn.Parameter(
                    torch.zeros(1, self.buffer_size_text, decoder_embed_dim)
                )

        # ========= Decoder Blocks =========
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(decoder_depth)
            ]
        )

        # ========= Decoder Norm =========
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # ========= Diffusion Temporal and Spatial Embedding =========
        self.diffusion_temporal_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, decoder_embed_dim)
        )
        self.diffusion_spatial_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        # ========= Initialize Weights =========
        # 先初始化所有参数
        self.initialize_weights()
        
        # ========= Copy Parameters from Original to Local Causal =========
        # 复制训练好的参数到local causal blocks
        self.copy_encoder_parameters()
        
        # ========= Re-initialize Only New Parameters =========
        # 重新初始化新添加的融合网络参数
        self.initialize_fusion_weights()

        # ========= Video Diffusion Loss =========
        self.predict_video = predict_video
        if self.predict_video:
            # ========= Video Diffusion Loss =========
            self.diffloss = DiffLoss(
                target_channels=self.token_embed_dim,
                z_channels=decoder_embed_dim,
                width=diffloss_w,
                depth=diffloss_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                n_frames=self.n_frames,
                language_emb_model=self.language_emb_model,
                language_emb_model_type=self.language_emb_model_type,
            )

            # ========= Wrist Video Diffusion Loss =========
            if self.predict_wrist_img:
                self.diffloss_wrist = DiffLoss(
                    target_channels=self.token_embed_dim,
                    z_channels=decoder_embed_dim,
                    width=diffloss_w,
                    depth=diffloss_d,
                    num_sampling_steps=num_sampling_steps,
                    grad_checkpointing=grad_checkpointing,
                    n_frames=self.n_frames,
                    language_emb_model=self.language_emb_model,
                    language_emb_model_type=self.language_emb_model_type,
                )

        # ========= Action Diffusion Loss =========
        if self.predict_action:
            self.diffactloss = DiffActLoss(
                target_channels=act_dim,
                z_channels=decoder_embed_dim,
                width=diffloss_act_w,
                depth=diffloss_act_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                n_frames=self.n_frames,
                act_model_type=action_model_params["act_model_type"],
                act_diff_training_steps=act_diff_training_steps,
                act_diff_testing_steps=act_diff_testing_steps,
                num_attention_heads=action_model_params.get("num_attention_heads", 8), #new
                language_emb_model=self.language_emb_model,
                language_emb_model_type=self.language_emb_model_type,
            )

        
        # ========= Proprioception Diffusion Loss =========
        if self.predict_proprioception:
            if self.task_name == "umi":
                self.diffproploss = DiffActLoss(
                    target_channels=6,
                    z_channels=decoder_embed_dim,
                    width=diffloss_act_w,
                    depth=diffloss_act_d,
                    num_sampling_steps=num_sampling_steps,
                    grad_checkpointing=grad_checkpointing,
                    n_frames=self.n_frames,
                    act_model_type=action_model_params["act_model_type"],
                    act_diff_training_steps=act_diff_training_steps,
                    act_diff_testing_steps=act_diff_testing_steps,
                    language_emb_model=self.language_emb_model,
                    language_emb_model_type=self.language_emb_model_type,
                )
            elif self.task_name == 'toolhang':
                self.diffproploss = DiffActLoss(
                        target_channels=9,
                        z_channels=decoder_embed_dim,
                        width=diffloss_act_w,
                        depth=diffloss_act_d,
                        num_sampling_steps=num_sampling_steps,
                        grad_checkpointing=grad_checkpointing,
                        n_frames=self.n_frames,
                        act_model_type=action_model_params["act_model_type"],
                        act_diff_training_steps=act_diff_training_steps,
                        act_diff_testing_steps=act_diff_testing_steps,
                        language_emb_model=self.language_emb_model,
                        language_emb_model_type=self.language_emb_model_type,
                    )
            else:
                raise NotImplementedError
            

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.fake_latent_x, std=0.02)
        torch.nn.init.normal_(self.fake_action_latent, std=0.02)

        if self.predict_wrist_img:
            torch.nn.init.normal_(self.fake_latent_wrist_x, std=0.02)

        if self.use_history_action:
            torch.nn.init.normal_(self.fake_latent_history_action, std=0.02)

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                torch.nn.init.normal_(self.fake_latent, std=0.02)

        torch.nn.init.normal_(self.temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.spatial_pos_embed, std=0.02)

        torch.nn.init.normal_(self.decoder_temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_spatial_pos_embed, std=0.02)

        torch.nn.init.normal_(self.diffusion_temporal_embed, std=0.02)
        torch.nn.init.normal_(self.diffusion_spatial_embed, std=0.02)

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                torch.nn.init.normal_(self.text_pos_embed, std=0.02)
                torch.nn.init.normal_(self.decoder_text_pos_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def copy_encoder_parameters(self):
        """Copy parameters from original encoder blocks to local causal encoder blocks"""
        for i, (orig_block, local_block) in enumerate(zip(self.encoder_blocks, self.local_causal_encoder_blocks)):
            # Copy attention parameters
            local_block.attn.qkv.weight.data = orig_block.attn.qkv.weight.data.clone()
            if orig_block.attn.qkv.bias is not None:
                local_block.attn.qkv.bias.data = orig_block.attn.qkv.bias.data.clone()
            
            local_block.attn.proj.weight.data = orig_block.attn.proj.weight.data.clone()
            if orig_block.attn.proj.bias is not None:
                local_block.attn.proj.bias.data = orig_block.attn.proj.bias.data.clone()
            
            # Copy MLP parameters
            local_block.mlp.fc1.weight.data = orig_block.mlp.fc1.weight.data.clone()
            if orig_block.mlp.fc1.bias is not None:
                local_block.mlp.fc1.bias.data = orig_block.mlp.fc1.bias.data.clone()
            
            local_block.mlp.fc2.weight.data = orig_block.mlp.fc2.weight.data.clone()
            if orig_block.mlp.fc2.bias is not None:
                local_block.mlp.fc2.bias.data = orig_block.mlp.fc2.bias.data.clone()
            
            # Copy normalization parameters
            local_block.norm1.weight.data = orig_block.norm1.weight.data.clone()
            local_block.norm1.bias.data = orig_block.norm1.bias.data.clone()
            
            local_block.norm2.weight.data = orig_block.norm2.weight.data.clone()
            local_block.norm2.bias.data = orig_block.norm2.bias.data.clone()
        
        # Copy encoder norm parameters
        self.local_causal_encoder_norm.weight.data = self.encoder_norm.weight.data.clone()
        self.local_causal_encoder_norm.bias.data = self.encoder_norm.bias.data.clone()

    def initialize_fusion_weights(self):
        """Initialize only the fusion network weights, preserving copied parameters"""
        # 只初始化融合网络的参数
        for m in self.feature_fusion.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.device).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, t, seq_len, embed_dim = x.shape

        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, t, seq_len, device=x.device)
        # Create the spatial mask for one frame (t=1)
        spatial_mask = torch.zeros(bsz, seq_len, device=x.device)
        spatial_mask = torch.scatter(
            spatial_mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        # Expand the spatial mask to all frames in the video
        mask = spatial_mask.unsqueeze(1).expand(-1, t, -1)

        return mask

    def forward_mae_encoder(
        self,
        x,
        mask,
        cond,
        text_latents=None,
        history_nactions=None,
        nactions=None,
        task_mode=None,
        proprioception_input={},
    ):
        B, T, S, _ = x.size()
        mask = rearrange(mask, "b t s -> b (t s)")

        # ========= Mask Input =========
        if task_mode == "policy_model":
            cond = self.z_proj_cond(cond)
            cond = rearrange(cond, "b t s c -> b (t s) c")
            x = self.fake_latent_x.unsqueeze(1).expand(B, cond.size(1), -1)

            if self.predict_wrist_img:
                wrist_x = self.fake_latent_wrist_x.unsqueeze(1).expand(
                    B, cond.size(1), -1
                )

        elif task_mode == "inverse_model":
            x = self.z_proj(x)
            x = rearrange(x, "b t s c -> b (t s) c")

            if self.predict_wrist_img:
                wrist_x = self.z_proj_wrist(proprioception_input["pred_second_image_z"])
                wrist_x = rearrange(wrist_x, "b t s c -> b (t s) c")

            cond = self.fake_latent_x.unsqueeze(1).expand(B, x.size(1), -1)

        else:
            cond = self.z_proj_cond(cond)
            cond = rearrange(cond, "b t s c -> b (t s) c")

            x = self.z_proj(x)
            x = rearrange(x, "b t s c -> b (t s) c")
            fake_latent_expanded = self.fake_latent_x.unsqueeze(1).expand(
                B, x.size(1), -1
            )
            x[mask == 1] = fake_latent_expanded[mask == 1].to(x.dtype)

            if self.predict_wrist_img:
                wrist_x = self.z_proj_wrist(proprioception_input["pred_second_image_z"])
                wrist_x = rearrange(wrist_x, "b t s c -> b (t s) c")
                fake_wrist_latent_expanded = self.fake_latent_wrist_x.unsqueeze(
                    1
                ).expand(B, wrist_x.size(1), -1)
                wrist_x[mask == 1] = fake_wrist_latent_expanded[mask == 1].to(
                    wrist_x.dtype
                )

        embed_dim = cond.size(2)

        # ========= History Action =========
        if self.use_history_action:
            if history_nactions is None:
                history_action_latents = self.fake_latent_history_action.unsqueeze(
                    0
                ).repeat(B, T * self.n_frames, 1)
            else:
                history_action_latents = self.history_action_proj_cond(history_nactions)

                if self.training:
                    history_action_mask = (
                        torch.rand(B, T * self.n_frames) > self.action_mask_ratio
                    ).int()
                    history_action_latents[history_action_mask == 1] = (
                        self.fake_latent_history_action.to(history_action_latents.dtype)
                    )

            history_action_latents_expand = history_action_latents.repeat_interleave(
                self.buffer_size_action, dim=1
            )

        # ========= Proprioception =========
        if self.use_proprioception:
            if self.task_name == "umi":
                proprioception_state_cond = torch.cat(
                    [
                        proprioception_input["robot0_eef_pos"],
                        proprioception_input["robot0_eef_rot_axis_angle"],
                        proprioception_input["robot0_gripper_width"],
                        proprioception_input["robot0_eef_rot_axis_angle_wrt_start"],
                    ],
                    dim=-1,
                )
                proprioception_state_cond = self.proprioception_proj_cond(
                    proprioception_state_cond.float()
                )
                proprioception_state_cond_expand = (
                    proprioception_state_cond.repeat_interleave(
                        self.buffer_size_properception, dim=1
                    )
                )
            else:
                proprioception_image_cond = self.proprioception_image_proj_cond(
                    proprioception_input["second_image_z"]
                )
                proprioception_image_cond = rearrange(
                    proprioception_image_cond, "b t s c -> b (t s) c"
                )

                proprioception_state_cond = torch.cat(
                    [
                        proprioception_input["robot0_eef_pos"],
                        proprioception_input["robot0_eef_quat"],
                        proprioception_input["robot0_gripper_qpos"],
                    ],
                    dim=-1,
                )
                proprioception_state_cond = self.proprioception_proj_cond(
                    proprioception_state_cond
                )
                proprioception_state_cond_expand = (
                    proprioception_state_cond.repeat_interleave(
                        self.buffer_size_properception, dim=1
                    )
                )

        # ========= Action =========
        if task_mode == "dynamic_model":
            action_latents = self.action_proj_cond(nactions)
        else:
            action_latents = self.fake_action_latent.unsqueeze(0).repeat(B, 16, 1)
        action_latents_expand = action_latents.repeat_interleave(
            self.buffer_size_action, dim=1
        )

        # ========= Wrist Video =========
        if self.predict_wrist_img:
            parts = [x, wrist_x, cond]
            if self.use_history_action:
                parts.append(history_action_latents_expand)
            parts.append(action_latents_expand)
            if self.use_proprioception:
                parts.extend(
                    [proprioception_image_cond, proprioception_state_cond_expand]
                )
            x = torch.cat(parts, dim=-1)
        else:
            parts = [x, cond]
            if self.use_history_action:
                parts.append(history_action_latents_expand)
            parts.append(action_latents_expand)

            if self.use_proprioception:
                if self.task_name == "umi":
                    parts.append(proprioception_state_cond_expand)
                else:
                    parts.extend(
                        [proprioception_image_cond, proprioception_state_cond_expand]
                    )
            x = torch.cat(parts, dim=-1)

        # ========= Projection =========
        x = self.proj_cond_x_layer(x)

        # ========= Position Embedding =========
        temporal_pos_embed_expanded = self.temporal_pos_embed.unsqueeze(2).expand(
            -1, -1, S, -1
        ) 
        spatial_pos_embed_expanded = self.spatial_pos_embed.unsqueeze(1).expand(
            -1, T, -1, -1
        ) 

        combined_pos_embed = (
            temporal_pos_embed_expanded + spatial_pos_embed_expanded
        ).reshape(-1, T * S, embed_dim)
        x = x + combined_pos_embed

        # ========= Language Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                text_latents = text_latents.unsqueeze(1).repeat(
                    1, self.buffer_size_text, 1
                )

                ## this is for cfg
                if self.training:
                    drop_latent_mask = torch.rand(B) < self.label_drop_prob
                    drop_latent_mask = (
                        drop_latent_mask.unsqueeze(-1).to(self.device).to(x.dtype)
                    )
                    drop_latent_mask = drop_latent_mask.unsqueeze(1).repeat(
                        1, self.buffer_size_text, 1
                    )
                    text_latents = (
                        drop_latent_mask
                        * self.fake_latent.unsqueeze(1).repeat(
                            1, self.buffer_size_text, 1
                        )
                        + (1 - drop_latent_mask) * text_latents
                    )

                text_latents = text_latents + self.text_pos_embed
                x = torch.cat([text_latents, x], dim=1)

        # ========= Normalization =========
        x = self.z_proj_ln(x)

        # ========= Transformer Encoder Blocks =========
        # Original transformer processing
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        global_features = self.encoder_norm(x)

        # ========= Local Causal Transformer Processing =========
        # Process with local causal attention (copy of original processing)
        local_x = x.clone()  # Start with the same input
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.local_causal_encoder_blocks:
                local_x = checkpoint(block, local_x)
        else:
            for block in self.local_causal_encoder_blocks:
                local_x = block(local_x)
        local_features = self.local_causal_encoder_norm(local_x)
        
        # ========= Feature Fusion =========
        # 现在使用真正的local attention特征进行融合
        lambda_local = self.lambda_local
        
        # 残差连接：global_features + small_adjustment
        adjustment = self.feature_fusion(
            torch.cat([global_features, local_features], dim=-1)
        )
        
        # 使用很小的权重来避免破坏原始特征
        fused_x = global_features + lambda_local * adjustment

        return fused_x

    def forward_mae_decoder(self, x, mask):
        B, T, S = mask.size()
        mask = rearrange(mask, "b t s -> b (t s)")
        x = self.decoder_embed(x)
        _, _, embed_dim = x.shape

        # ========= Position Embedding =========
        decoder_temporal_pos_embed_expanded = self.decoder_temporal_pos_embed.unsqueeze(
            2
        ).expand(
            -1, -1, S, -1
        ) 
        decoder_spatial_pos_embed_expanded = self.decoder_spatial_pos_embed.unsqueeze(
            1
        ).expand(
            -1, T, -1, -1
        ) 
        decoder_combined_pos_embed = (
            decoder_temporal_pos_embed_expanded + decoder_spatial_pos_embed_expanded
        ).reshape(1, T * S, embed_dim)

        # ========= Language Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                combined_pos_embed = torch.cat(
                    [self.decoder_text_pos_embed, decoder_combined_pos_embed], dim=1
                )
            else:
                combined_pos_embed = decoder_combined_pos_embed
        else:
            combined_pos_embed = decoder_combined_pos_embed

        x = x + combined_pos_embed

        # ========= Transformer Decoder Blocks =========
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        # ========= Language Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                x = x[:, self.buffer_size_text :]

        # ========= Diffusion Position Embedding =========
        diffusion_temporal_pos_embed_expanded = self.diffusion_temporal_embed.unsqueeze(
            2
        ).expand(
            -1, -1, S, -1
        )
        diffusion_spatial_pos_embed_expanded = self.diffusion_spatial_embed.unsqueeze(
            1
        ).expand(
            -1, T, -1, -1
        )
        diffusion_combined_pos_embed = (
            diffusion_temporal_pos_embed_expanded + diffusion_spatial_pos_embed_expanded
        ).reshape(1, T * S, embed_dim)

        x = x + diffusion_combined_pos_embed

        return x

    def forward_loss(
        self,
        z,
        target,
        mask,
        nactions=None,
        task_mode=None,
        gt_wrist_latents=None,
        gt_properception=None,
        text_latents=None,
    ):
        if task_mode == "video_model" or task_mode == "dynamic_model":
            if self.predict_wrist_img:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents
                )
                video_loss_wrist = self.diffloss_wrist(
                    z=z, target=gt_wrist_latents, mask=mask, text_latents=text_latents
                )
                video_loss = video_loss + video_loss_wrist
            else:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents
                )

            act_loss = torch.tensor(0.0).to(self.device)
            loss = video_loss

        elif task_mode == "policy_model" or task_mode == "inverse_model":
            act_loss = self.diffactloss(
                z=z, target=nactions, task_mode=task_mode, text_latents=text_latents
            )
            video_loss = torch.tensor(0.0).to(self.device)
            loss = act_loss

        elif task_mode == "full_dynamic_model":
            if self.predict_wrist_img:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents
                )
                video_loss_wrist = self.diffloss_wrist(
                    z=z, target=gt_wrist_latents, mask=mask, text_latents=text_latents
                )
                video_loss = video_loss + video_loss_wrist
            else:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents
                )
            act_loss = self.diffactloss(
                z=z, target=nactions, task_mode=task_mode, text_latents=text_latents
            )
            loss = video_loss + act_loss

        if self.predict_proprioception:
            properception_loss = self.diffproploss(
                z=z, target=gt_properception, text_latents=text_latents
            )
            loss = loss + properception_loss

        return loss, video_loss, act_loss

    def forward(
        self,
        imgs,
        cond,
        history_nactions=None,
        nactions=None,
        text_latents=None,
        task_mode=None,
        proprioception_input={},
    ):
        self.device = cond.device
        B, T, C, H, W = imgs.size()

        # ========= Patchify =========
        imgs = rearrange(
            imgs, "b t c h w -> (b t) c h w"
        )
        x = self.patchify(imgs)
        x = rearrange(x, "(b t) seq_len c -> b t seq_len c", b=B)

        cond = rearrange(cond, "b t c h w -> (b t) c h w")
        cond = self.patchify(cond)
        cond = rearrange(
            cond, "(b t) seq_len c -> b t seq_len c", b=B
        )

        # ========= Proprioception =========
        if self.use_proprioception:
            if "second_image_z" in proprioception_input:
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"], "b t c h w -> (b t) c h w"
                )
                proprioception_input["second_image_z"] = self.patchify(
                    proprioception_input["second_image_z"]
                )
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        # ========= Predicted Wrist Image =========
        if self.predict_wrist_img:
            if "pred_second_image_z" in proprioception_input:
                proprioception_input["pred_second_image_z"] = rearrange(
                    proprioception_input["pred_second_image_z"],
                    "b t c h w -> (b t) c h w",
                )
                proprioception_input["pred_second_image_z"] = self.patchify(
                    proprioception_input["pred_second_image_z"]
                )
                proprioception_input["pred_second_image_z"] = rearrange(
                    proprioception_input["pred_second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        if text_latents is not None and hasattr(self, "text_proj_cond"):
            if self.language_emb_model_type == 1:
                text_latents = self.text_proj_cond(text_latents)

        gt_latents = x.clone().detach()

        # ========= Predicted Wrist Image =========
        if self.predict_wrist_img:
            if "pred_second_image_z" in proprioception_input:
                gt_wrist_latents = (
                    proprioception_input["pred_second_image_z"].clone().detach()
                )
                gt_wrist_latents = rearrange(
                    gt_wrist_latents, "b t s c -> b (t s) c"
                )

        # ========= Sample Orders =========
        orders = self.sample_orders(bsz=B)
        mask = self.random_masking(x, orders)  # [1, 4, 256]

        # ========= MAE Encoder =========
        x = self.forward_mae_encoder(
            x,
            mask,
            cond,
            text_latents,
            history_nactions,
            nactions,
            task_mode=task_mode,
            proprioception_input=proprioception_input,
        )

        # ========= MAE Decoder =========
        z = self.forward_mae_decoder(x, mask)

        # ========= Diffloss over Video and Action =========
        mask = rearrange(mask, "b t s -> b (t s)")
        gt_latents = rearrange(
            gt_latents, "b t s c -> b (t s) c"
        )

        # ========= Predict Proprioception =========
        if self.predict_proprioception:
            if self.task_name == "umi":
                gt_properception = proprioception_input[
                    "robot0_eef_rot_axis_angle_wrt_start_pred"
                ]
            elif self.task_name == "toolhang":
                gt_properception = torch.cat([proprioception_input['robot0_eef_pos_pred'], 
                                              proprioception_input['robot0_eef_quat_pred'], 
                                              proprioception_input['robot0_gripper_qpos_pred']], 
                                             dim=-1)
            else:
                raise NotImplementedError

            if self.predict_wrist_img:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_wrist_latents=gt_wrist_latents,
                    gt_properception=gt_properception,
                    text_latents=text_latents,
                )
            else:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_properception=gt_properception,
                    text_latents=text_latents,
                )
        else:
            if self.predict_wrist_img:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_wrist_latents=gt_wrist_latents,
                    text_latents=text_latents,
                )
            else:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    text_latents=text_latents,
                )

        return loss, video_loss, act_loss

    def sample_tokens(
        self,
        bsz,
        cond,
        text_latents=None,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        temperature=1.0,
        progress=False,
        history_nactions=None,
        nactions=None,
        proprioception_input={},
        task_mode=None,
        vae_model=None,
        x=None,
    ):
        self.device = cond.device
        B, T, C, H, W = cond.size()
        cond = rearrange(cond, "b t c h w -> (b t) c h w")
        cond = self.patchify(cond)
        cond = rearrange(
            cond, "(b t) seq_len c -> b t seq_len c", b=B
        )

        # ========= Proprioception =========
        if self.use_proprioception:
            if "second_image_z" in proprioception_input:
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"], "b t c h w -> (b t) c h w"
                )
                proprioception_input["second_image_z"] = self.patchify(
                    proprioception_input["second_image_z"]
                )
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        if text_latents is not None and hasattr(self, "text_proj_cond"):
            if self.language_emb_model_type == 1:
                text_latents = self.text_proj_cond(text_latents)

        # ========= Mask =========
        if task_mode == "inverse_model":
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.patchify(x)
            tokens = rearrange(
                x, "(b t) seq_len c -> b t seq_len c", b=B
            )
            mask = torch.zeros(bsz, self.n_frames, self.seq_len).to(self.device)
        else:
            # init and sample generation orders
            tokens = torch.zeros(
                bsz, self.n_frames, self.seq_len, self.token_embed_dim
            ).to(self.device)
            mask = torch.ones(bsz, self.n_frames, self.seq_len).to(self.device)
            if self.predict_wrist_img:
                proprioception_input["pred_second_image_z"] = torch.zeros(
                    bsz, self.n_frames, self.seq_len, self.token_embed_dim
                ).to(self.device)

        # ========= Sample Orders =========
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # ========= Predict Video =========
        if self.predict_video:
            for step in indices:
                cur_tokens = tokens.clone()

                if self.predict_wrist_img:
                    cur_wrist_tokens = proprioception_input[
                        "pred_second_image_z"
                    ].clone()

                x = self.forward_mae_encoder(
                    tokens,
                    mask,
                    cond,
                    text_latents,
                    history_nactions=history_nactions,
                    nactions=nactions,
                    task_mode=task_mode,
                    proprioception_input=proprioception_input,
                )
                z = self.forward_mae_decoder(x, mask)

                if self.predict_action:
                    act_cfg = 1.0
                    sampled_token_latent_act = self.diffactloss.sample(
                        z, temperature, cfg=act_cfg, text_latents=text_latents
                    )
                else:
                    sampled_token_latent_act = None

                # ========= Predict action and return if task_mode is inverse_model or policy_model=========
                if task_mode == "inverse_model" or task_mode == "policy_model":
                    return None, sampled_token_latent_act

                # ========= Mask Ratio =========
                # mask ratio for the next round, following MaskGIT and MAGE.
                mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
                mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(
                    self.device
                )

                # take the first frame mask
                mask_ = mask[:, 0]

                # masks out at least one for the next iteration
                mask_len = torch.maximum(
                    torch.Tensor([1]).to(self.device),
                    torch.minimum(
                        torch.sum(mask_, dim=-1, keepdims=True) - 1, mask_len
                    ),
                )

                # get masking for next iteration and locations to be predicted in this iteration
                mask_next = mask_by_order(
                    mask_len[0], orders, bsz, self.seq_len, self.device
                )

                ## expand mask_next to all frames
                mask_next = mask_next.unsqueeze(1).expand(-1, T, -1)
                mask_next = rearrange(mask_next, "b t s -> b (t s)")
                mask = rearrange(mask, "b t s -> b (t s)")

                if step >= num_iter - 1:
                    mask_to_pred = mask[:bsz].bool()
                else:
                    mask_to_pred = torch.logical_xor(
                        mask[:bsz].bool(), mask_next.bool()
                    )
                mask = mask_next
                mask = rearrange(mask, "b (t s) -> b t s", t=self.n_frames)

                if not cfg == 1.0:
                    mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

                # sample token latents for this step
                z = z[mask_to_pred.nonzero(as_tuple=True)]
                # cfg schedule follow Muse
                if cfg_schedule == "linear":
                    cfg_iter = (
                        1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
                    )
                elif cfg_schedule == "constant":
                    cfg_iter = cfg
                else:
                    raise NotImplementedError

                sampled_token_latent = self.diffloss.sample(
                    z, temperature, cfg_iter, text_latents=text_latents
                )

                if not cfg == 1.0:
                    sampled_token_latent, _ = sampled_token_latent.chunk(
                        2, dim=0
                    )  # Remove null class samples
                    mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

                cur_tokens = rearrange(cur_tokens, "b t s c -> b (t s) c")
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
                cur_tokens = rearrange(
                    cur_tokens, "b (t s) c -> b t s c", t=self.n_frames
                )
                tokens = cur_tokens.clone()

                # ========= Predict Wrist Image =========
                if self.predict_wrist_img:
                    sampled_wrist_token_latent = self.diffloss_wrist.sample(
                        z, temperature, cfg_iter, text_latents=text_latents
                    )

                    if not cfg == 1.0:
                        sampled_wrist_token_latent, _ = (
                            sampled_wrist_token_latent.chunk(2, dim=0)
                        )  # Remove null class samples

                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b t s c -> b (t s) c"
                    )
                    cur_wrist_tokens[mask_to_pred.nonzero(as_tuple=True)] = (
                        sampled_wrist_token_latent
                    )
                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b (t s) c -> b t s c", t=self.n_frames
                    )
                    proprioception_input["pred_second_image_z"] = (
                        cur_wrist_tokens.clone()
                    )

            # ========= Unpatchify =========
            tokens = rearrange(tokens, "b t s c -> (b t) s c")
            tokens = self.unpatchify(tokens)
            # tokens = rearrange(tokens, '(b t) c h w -> b t c h w', b=B)

            if self.predict_wrist_img:
                wrist_tokens = rearrange(
                    proprioception_input["pred_second_image_z"], "b t s c -> (b t) s c"
                )
                wrist_tokens = self.unpatchify(wrist_tokens)

        else:
            raise NotImplementedError

        if self.predict_wrist_img:
            return wrist_tokens, sampled_token_latent_act
        else:
            return tokens, sampled_token_latent_act


def mar_tiny(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=3,
        encoder_num_heads=6,
        decoder_embed_dim=768,
        decoder_depth=3,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_small(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=768,
        decoder_depth=6,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_embed_dim=1280,
        decoder_depth=20,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
