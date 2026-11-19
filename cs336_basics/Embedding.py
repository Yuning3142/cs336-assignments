import torch
import torch.nn  as nn


class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device = None,dtype=None):
        super().__init__()
