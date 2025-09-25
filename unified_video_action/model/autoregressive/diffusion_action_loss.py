import torch
import torch.nn as nn
from einops import rearrange

from unified_video_action.model.autoregressive.diffusion import create_diffusion
from unified_video_action.model.autoregressive.diffusion_loss import SimpleMLPAdaLN
#from unified_video_action.model.autoregressive.cross_attention_diffusion import CrossAttentionAdaLN


#实现基于扩散模型的动作预测损失函数的主要代码，用于从视频特征预测动作序列
class DiffActLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(
        self,
        target_channels, #目标动作的通道数
        z_channels, #输入特征的通道数
        depth,  #MLP的网络深度
        width,  #MLP的宽度
        num_sampling_steps,  #采样步数
        grad_checkpointing=False,  #梯度检查点(为了优化内存的)
        n_frames=4, #输入帧数
        act_diff_training_steps=1000, #训练时的扩散步数
        act_diff_testing_steps="100", #测试时的扩散步数
        act_model_type="conv_fc",  #特征处理架构类型（卷积+全连接）
        num_attention_heads=8, #new
        **kwargs
    ):
        super(DiffActLoss, self).__init__()
        self.in_channels = target_channels
        self.n_frames = n_frames

        self.language_emb_model = kwargs["language_emb_model"]
        self.language_emb_model_type = kwargs["language_emb_model_type"]

        self.act_model_type = act_model_type

        if self.act_model_type == "conv_fc":
            self.w = 16
            self.h = 16
            self.num_frames = 4 #输入帧数
            self.num_actions = 16 #输出动作数

            # CNN层：Single convolutional layer for spatial processing 处理空间特征，提取视觉模式
            self.conv = nn.Sequential(
                nn.Conv2d(z_channels, z_channels, kernel_size=3, stride=1, padding=1), #卷积层（16*16）
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # (B×4, 1024, 4, 4)  ← 16个空间位置的特征
            )

            # FC层：Fully connected layer for action latent prediction 进行特征映射和维度变换
            self.fc = nn.Sequential(
                nn.Linear(z_channels * 4 * 4, z_channels), #拉平 (B×4, 16384)   ← 一维向量 
                nn.ReLU(),
                nn.Linear(z_channels, z_channels),  # Predict latents for all actions: (B×4, 1024) ← 全局语义特征
            )

            # 时间插值层：处理时间维度的扩展
            self.interpolate = nn.Linear(self.num_frames, self.num_actions) #输入: 4个时间步的特征  输出: 16个动作序列的特征

            # 精炼层：最终的特征优化
            self.refine = nn.Sequential(
                nn.Linear(z_channels, z_channels),
                nn.ReLU(),
                nn.Linear(z_channels, z_channels),
            )

        # 3D卷积，同时处理时空特征
        elif self.act_model_type == "conv_ori":
            self.w = 16 # 空间宽度
            self.h = 16 # 空间高度
            self.conv_transpose3d = nn.ConvTranspose3d(
                in_channels=z_channels,
                out_channels=z_channels,
                kernel_size=(4, 1, 1), 
                stride=(4, 1, 1),
            )
            self.avg_pool = nn.AvgPool3d(kernel_size=(1, self.w, self.h))
            
        # 轻量级的1D卷积网络，主要用于快速处理时序特征
        elif self.act_model_type == 'conv2':
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(in_channels=256, out_channels=16, kernel_size=7, padding=3)
            )
        
        # 只需要进行维度映射
        elif self.act_model_type == 'fc2':
            self.fc = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),  # Add an activation function (optional, but common practice)
                nn.Linear(256, 16)
            )
            
        # Cross attention model - no preprocessing needed
        elif self.act_model_type == 'cross_attention':
            pass
            
        else:
            raise NotImplementedError

        # 强制使用SimpleMLPAdaLN以保持预训练参数
        # 暂时不使用CrossAttentionAdaLN，因为参数是初始化的会影响效果
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

        self.train_diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="cosine",
            diffusion_steps=act_diff_training_steps,
        )
        self.gen_diffusion = create_diffusion(
            timestep_respacing=act_diff_testing_steps, noise_schedule="cosine"
        )

    def forward(self, target, z, task_mode=None, text_latents=None):
        #1. 获取输入维度
        bsz, seq_len, _ = target.shape 
        #2. 根据 act_model_type 预处理特征 z
        if self.act_model_type == "conv_fc":
            z = rearrange(z, "b (t s) c -> (b t) s c", t=self.n_frames)
            z = rearrange(z, "b (w h) c -> b w h c", w=self.w)
            z = rearrange(z, "b w h c -> b c w h")
            z = self.conv(z)
            z = rearrange(z, "b c w h -> b (c w h)")
            z = self.fc(z)

            z = rearrange(z, "(b t) c -> b t c", t=self.n_frames)
            z = z.permute(0, 2, 1)
            z = self.interpolate(z)
            z = z.permute(0, 2, 1)
            z = self.refine(z)
            
        elif self.act_model_type == "conv_ori":
            z = rearrange(
                z, "b (t s) c -> b t s c", t=self.n_frames
            )
            z = rearrange(
                z, "b t (w h) c -> b c t w h", w=self.w
            )
            z = self.conv_transpose3d(z)
            z = self.avg_pool(z)
            z = rearrange(z, "b c t w h -> b (t w h) c")
            
        elif self.act_model_type == 'conv2':
            z = self.conv(z)
        
        elif self.act_model_type == 'fc2':
            z = self.fc(z.transpose(1, 2))
            z = z.transpose(1, 2)
            
        elif self.act_model_type == 'cross_attention':
            # 现在强制使用SimpleMLPAdaLN，不需要特殊预处理
            pass
            
        else:
            raise NotImplementedError

        #3. 重塑特征，准备进行扩散模型的训练
        target = target.reshape(bsz * seq_len, -1) # 展平动作序列
        z = z.reshape(bsz * seq_len, -1) # 展平特征序列

        #4. 随机选择扩散步数
        t = torch.randint(
            0,
            self.train_diffusion.num_timesteps,
            (target.shape[0],),
            device=target.device,
        )

        #5. 计算扩散损失
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(
            self.net, target, t, model_kwargs
        )

        #6. 计算最终损失
        action_loss = loss_dict["loss"].reshape(bsz, seq_len)
        total_loss = torch.mean(action_loss)

        return total_loss

    def sample(self, z, temperature=1.0, cfg=1.0, text_latents=None):
        if self.act_model_type == "conv_fc":
            z = rearrange(z, "b (t s) c -> (b t) s c", t=self.n_frames)
            z = rearrange(z, "b (w h) c -> b w h c", w=self.w)
            z = rearrange(z, "b w h c -> b c w h")
            z = self.conv(z)
            z = rearrange(z, "b c w h -> b (c w h)")
            z = self.fc(z)

            z = rearrange(z, "(b t) c -> b t c", t=self.n_frames)
            z = z.permute(0, 2, 1)
            z = self.interpolate(z)
            z = z.permute(0, 2, 1)
            z = self.refine(z)
            
        elif self.act_model_type == "conv_ori":
            z = rearrange(
                z, "b (t s) c -> b t s c", t=self.n_frames
            )
            z = rearrange(
                z, "b t (w h) c -> b c t w h", w=self.w
            )
            z = self.conv_transpose3d(z)
            z = self.avg_pool(z)
            z = rearrange(z, "b c t w h -> b (t w h) c")
        
        elif self.act_model_type == 'conv2':
            z = self.conv(z)
            
        elif self.act_model_type == 'fc2':
            z = self.fc(z.transpose(1, 2))
            z = z.transpose(1, 2)
            
        elif self.act_model_type == 'cross_attention':
            # 现在强制使用SimpleMLPAdaLN，不需要特殊预处理
            pass
            
        else:
            raise NotImplementedError

        bsz, seq_len, _ = z.shape
        z = rearrange(z, "b t c -> (b t) c")


        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=False,
            temperature=temperature,
        )

        sampled_token_latent = rearrange(
            sampled_token_latent, "(b t) c -> b t c", b=bsz
        )
        return sampled_token_latent
