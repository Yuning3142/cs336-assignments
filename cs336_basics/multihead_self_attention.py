import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.RoPE import RoPE

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,max_seq_len,theta):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model,d_model)
        self.k_proj = Linear(d_model,d_model)
        self.v_proj = Linear(d_model,d_model)
        self.output_proj = Linear(d_model,d_model)
        self.Rope = RoPE(theta,self.head_dim,max_seq_len)

    def forward(self, x, mask=None,token_position=None):
        B,T,D = x.shape
        if token_position is None:
            token_position = torch.arange(T,device=x.device)
        Q = self.Rope(self.q_proj(x).reshape(B,T, self.num_heads, self.head_dim).transpose(1, 2),token_position)
        K = self.Rope(self.k_proj(x).reshape(B,T, self.num_heads, self.head_dim).transpose(1, 2),token_position)
        V = self.v_proj(x).reshape(B,T, self.num_heads, self.head_dim).transpose(1, 2)

        if mask is None:
            mask = torch.tril(torch.ones(T,T,dtype = torch.bool,device=x.device), diagonal=0)
        scaled_dot = scaled_dot_product_attention(Q, K, V, mask)
        scaled_dot = scaled_dot.transpose(1, 2).reshape(B,T,D)
        return self.output_proj(scaled_dot)







