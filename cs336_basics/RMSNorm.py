import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self,d_model,eps = 1e-5,device = None,dtype = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model,device = device, dtype = dtype))



    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        denominator = torch.sqrt(torch.mean(x**2,dim = -1,keepdim=True) + self.eps)
        x = torch.div(x,denominator)
        result = x * self.weight
        return result.to(in_dtype)


