import torch
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLM(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta,vocab_size,num_layers):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList(TransformerBlock(d_model,num_heads,d_ff,max_seq_len,theta) for _ in range(num_layers))
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model,vocab_size)

    def forward(self,x):
        x = self.token_embeddings(x)
        for transformer_block in self.layers:
            x = transformer_block(x)

        x = self.ln_final(x)
        return self.lm_head(x)

