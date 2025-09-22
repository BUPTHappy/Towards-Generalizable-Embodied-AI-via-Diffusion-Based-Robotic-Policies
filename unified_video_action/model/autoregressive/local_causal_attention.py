import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCausalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=5, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        #Multi-head attention
        self.attention = nn.MultiheadAttention(
          dim, num_heads, dropout=dropout, batch_first=True
        )

        #Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        #MLP
        self.mlp = nn.Sequential(
          nn.Linear(dim, dim * 4),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(dim * 4, dim),
          nn.Dropout(dropout),
        )
    
    #local causal mask
    def create_local_causal_mask(self, seq_len):
        mask = torch.zeros(seq_len, seq_len)

        for i in range (seq_len):
          #make it can only see current and previous window_size-1 positions
          start_idx = max(0, i - self.window_size + 1)
          mask[i, start_idx:i+1] = 1

        return mask==0
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # 添加边界检查
        if seq_len <= 0:
            return x
        
        #create local causal mask
        attn_mask = self.create_local_causal_mask(seq_len).to(x.device)
        
        # 确保mask在正确的设备上
        if attn_mask.device != x.device:
            attn_mask = attn_mask.to(x.device)

        #self-attention with local causal mask
        try:
            atten_out, attn_weights = self.attention(
              x, x, x, 
              attn_mask=attn_mask,
              need_weights=True
            )
        except RuntimeError as e:
            print(f"Attention error: {e}")
            print(f"Input shape: {x.shape}")
            print(f"Mask shape: {attn_mask.shape}")
            print(f"Device: {x.device}")
            raise e

        #residual connection
        x = self.norm1(x + atten_out)

        #MLP
        mpl_out = self.mlp(x)

        #residual connection
        x = self.norm2(x + mpl_out)

        return x  # 只返回x，不返回attn_weights
        