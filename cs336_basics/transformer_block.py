import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.silu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads,max_seq_len,theta)
        self.ffn = SwiGLU(d_model,d_ff)

    def forward(self,x,mask = None,token_postion = None):
        x = x + self.attn(self.ln1(x), mask, token_postion)
        return  x + self.ffn(self.ln2(x))

# model = TransformerBlock(4,2,3,4,5)
#
# print(model.state_dict().keys())