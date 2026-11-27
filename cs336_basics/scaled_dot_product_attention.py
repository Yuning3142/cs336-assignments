import torch
import math


def softmax(x, axis=-1):
    max_x = torch.max(x, axis=axis, keepdims=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x/torch.sum(exp_x,dim=axis,keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    K_new =  K.transpose(-1,-2)
    logits =  Q @ K_new/math.sqrt(Q.shape[-1])
    if mask is not None:
        logits = torch.masked_fill(logits, ~mask,float('-inf'))
    return torch.softmax(logits, dim=-1)  @ V





