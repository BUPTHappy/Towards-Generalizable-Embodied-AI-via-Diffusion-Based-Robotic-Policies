import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

# same as diffusion_loss.py
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
  

# ResBlock -> CrossAttentionBlock
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        #1. Cross-Attention : Q from action; K/V from video features
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        #2. Self-Attention : within action sequence
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        #3. MLP
        self.mlp = nn.Sequential(
          nn.Linear(dim, dim * 4, bias=True),
          nn.SiLU(),
          nn.Dropout(dropout),
          nn.Linear(dim * 4, dim,  bias=True),
          nn.Dropout(dropout),
        )

        #Layer normalization
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)

        #AdaLN modulation for cross-attention
        self.adaLN_modulation_cross = nn.Sequential(
          nn.SiLU(),
          nn.Linear(dim, 6 * dim, bias=True),
        )

        #AdaLN modulation for self-attention
        self.adaLN_modulation_self = nn.Sequential(
          nn.SiLU(),
          nn.Linear(dim, 6 * dim, bias=True),
        )

    
    def forward(self, x, video_features, c):
        """
          :param x: action sequence features [B, L, D]
          :param video_features: video features [B, L, D] 
          :param c: conditioning signal [B, D]
          :return: updated action features [B, L, D]
        """

        #1. Cross-Attention
        cross_mod = self.adaLN_modulation_cross(c)
        cross_scale, cross_shift, cross_gate = cross_mod.chunk(3, dim=1)

        x_norm = self.norm1(x)
        x_norm = x_norm * (1+ cross_scale.unsqueeze(1)) + cross_shift.unsqueeze(1)

        cross_out, _ = self.cross_attn(
          query = x_norm,
          key = video_features,
          value = video_features
        )

        x = x + cross_gate.unsqueeze(1) * cross_out

        #2. Self-Attention
        self_mod = self.adaLN_modulation_self(c)
        self_scale, self_shift, self_gate = self_mod.chunk(3, dim=1)

        x_norm = self.norm2(x)
        x_norm = x_norm * (1+ self_scale.unsqueeze(1)) + self_shift.unsqueeze(1)

        self_out, _ = self.self_attn(
          query = x_norm,
          key = x_norm,
          value = x_norm
        )
        
        x = x + self_gate.unsqueeze(1) * self_out

        #3. MLP
        mpl_mod = self.adaLN_modulation_self(c)
        mpl_scale, mpl_shift, mpl_gate = mpl_mod.chunk(3, dim=1)

        x_norm = self.norm3(x)
        x_norm = x_norm * (1+ mpl_scale.unsqueeze(1)) + mpl_shift.unsqueeze(1)

        mlp_out = self.mlp(x_norm)

        x = x + mpl_gate.unsqueeze(1) * mlp_out

        return x
      
      
# same as diffusion_loss.py
class FinalLayer(nn.Module):
    
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


#SimpleMLPAdaLN -> CrossAttentionAdaLN
class CrossAttentionAdaLN(nn.Module):

  def __init__(
    self,
    in_channels,
    model_channels,
    out_channels,
    z_channels,
    num_res_blocks,
    num_heads=8,
    grad_checkpointing=False,
  ):
      super().__init__()

      self.in_channels = in_channels
      self.model_channels = model_channels
      self.out_channels = out_channels
      self.num_res_blocks = num_res_blocks
      self.num_heads = num_heads
      self.grad_checkpointing = grad_checkpointing

      self.time_embed = TimestepEmbedder(model_channels)
      self.cond_embed = nn.Linear(z_channels, model_channels)
      self.input_proj = nn.Linear(in_channels, model_channels)
      self.video_proj = nn.Linear(z_channels, model_channels)

      cross_attn_blocks =[]
      for i in range(num_res_blocks):
        cross_attn_blocks.append(
          CrossAttentionBlock(model_channels, num_heads=num_heads)
        )

      self.cross_attn_blocks = nn.ModuleList(cross_attn_blocks)
      self.final_layer = FinalLayer(model_channels, out_channels)

      self.initialize_weights()

  def initialize_weights(self):
      for module in self.modules():
        if isinstance(module, nn.Linear):
          nn.init.xavier_uniform_(module.weight)
          if module.bias is not None:
            nn.init.constant_(module.bias, 0)
      
      for module in self.modules():
        if isinstance(module, nn.LayerNorm):
          nn.init.constant_(module.bias, 0)
          nn.init.constant_(module.weight, 1.0)
      
      nn.init.constant_(self.final_layer.linear.bias, 0)

  def forward(self, x ,t , c):
      x = self.input_proj(x)
      video_feature = self.video_proj(c)
      t_emb = self.time_embed(t)
      c_emb = self.cond_embed(c)

      y = t_emb + c_emb

      x = x.unsqueeze(1) #[N, 1, model_channels]
      video_feature = video_feature.unsqueeze(1) #[N, 1, model_channels]

      #Cross-Attention Block
      if self.grad_checkpointing and not torch.jit.is_scripting():
        for block in self.cross_attn_blocks:
          x = checkpoint(block, x, video_feature, y)
      else:
        for block in self.cross_attn_blocks:
          x = block(x, video_feature, y)

      x = x.squeeze(1) #[N, model_channels]
  
  def forward_with_cfg(self, x, t, c, cfg_scale):
      half = x[: len(x) // 2]
      combined = torch.cat([half, half], dim=0)
      model_out = self.forward(combined, t, c)
      eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
      cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
      half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
      eps = torch.cat([half_eps, half_eps], dim=0)
      return torch.cat([eps, rest], dim=1)

    