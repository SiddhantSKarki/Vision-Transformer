import torch
import torch.nn as nn
from torch.nn import functional as F


''' 
Transformer Encoder Code Starts Here
'''
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # No need to do masked attention because we're not doing anything autoregressive
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.value = nn.Linear(n_embd, head_size, bias=False)


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = torch.matmul(q, k.transpose(-2, -1)) * k.shape[-1]**0.5
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = torch.matmul(wei, v)

        return out, wei


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.attention_probs = None
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        attention_outs = [h(x) for h in self.heads]
        attention_out = torch.cat([output for output, _ in attention_outs], dim=-1)
        attention_out = self.proj(attention_out)
        attention_out = self.dropout(attention_out)
        self.attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outs], dim=1)
        return attention_out
        

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.multi_head = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, num_blocks, block_size):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(embedding_size, num_heads, block_size) for _ in range(num_blocks)]
        )
        
    def forward(self, x):
        return self.blocks(x)