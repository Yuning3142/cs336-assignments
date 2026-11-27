import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device = None):
        super().__init__()
        top = torch.arange(0,max_seq_len,device=device,dtype=torch.float32)
        down = torch.arange(0,d_k//2,device=device,dtype=torch.float32)
        feq = 1/torch.pow(theta,2 * down / d_k)
        angle = torch.outer(top,feq)
        self.register_buffer("cos_cache",torch.cos(angle),persistent=False)
        self.register_buffer("sin_cache", torch.sin(angle), persistent=False)


    def forward(self,x,token_position):

        first = x[..., 0::2]
        second = x[..., 1::2]
        token_cos = self.cos_cache[token_position]
        token_sin = self.sin_cache[token_position]
        first_rope = first * token_cos - second * token_sin
        sencond_rope =  first * token_sin + second * token_cos
        result = torch.empty_like(x)
        result[...,0::2] = first_rope
        result[...,1::2] = sencond_rope
        return  result




