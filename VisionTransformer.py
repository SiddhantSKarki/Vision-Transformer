# Copyright @Siddhant S. Karki 2024 

import torch
import torch.nn as nn
from torch.nn import functional as F

##### Hyper parameters#############
## TODO: Implement Config files instead of this
input_channels = 3 # 1 for Gray Scale Imgs and 3 for RGB
p_embd_size = 32
batch_size = 8
img_size = 256
kernel_size = 16
n_patches = kernel_size**2
precision = torch.float32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##### Hyper Parameters #############

#<-----------Transformer Encoder Code Starts Here-----------------------> 

#TODO: Implement Transformer Encoder Block

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.value = nn.Linear(n_embd, head_size, bias=False)


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = torch.matmul(q, k.transpose(-2, -1)) * k.shape[-1]**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = torch.matmul(wei, v)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
        

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
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.multi_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(embedding_size, num_heads) for _ in range(num_blocks)]
        )
        
    def forward(self, x):
        return self.blocks(x)

#<-----------Transformer Encoder Code Ends Here-----------------------> 
########################################################################
########################################################################
########################################################################
#<-----------Vision Transformer Code Starts Here----------------------->
class PatchEmbedding(nn.Module):
    def __init__(self,
                input_channels,
                embedding_size,
                patch_size,
                num_patches,
                precision,
                device):
        super().__init__()
        # Convert an image into patches
        self.sequence = nn.Sequential(
            nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=embedding_size,
                    kernel_size=patch_size,
                    stride=patch_size,
                    device=device,
                    dtype=precision
                ),
            nn.Flatten(2),
        )
        self.cls_tkn_embd = nn.Parameter(
                                    torch.randn(size=(batch_size, 1, embedding_size), device=device),
                                    requires_grad=True)
        self.pos_embd = nn.Parameter(
                                    torch.randn(size=(1,  num_patches + 1,  embedding_size), device=device),
                                    requires_grad=True)


    def forward(self, x):
        if x.dtype != precision:
            x = x.type(precision)
        x = self.sequence(x).permute(0, 2, 1)
        x = torch.cat([x, self.cls_tkn_embd], dim=1)
        x = x + self.pos_embd
        return x
    


class VisionTransformer(nn.Module):
    def __init__(self,
                input_channels,
                num_classes,
                embedding_size,
                patch_size,
                num_patches,
                num_heads,
                num_blocks,
                precision,
                device):
        super().__init__()

        self.emdeddings = PatchEmbedding(
                input_channels=input_channels,
                embedding_size=embedding_size,
                patch_size=patch_size,
                num_patches=num_patches,
                precision=precision,
                device=device)
        
        self.transformer_encoder = TransformerEncoder(
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_blocks=num_blocks
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4*embedding_size),
            nn.GELU(),
            nn.Linear(4*embedding_size, num_classes)
        )

        self.ln = nn.LayerNorm()


    def forward(self, x):
        x = self.emdeddings(x)
        x = self.transformer_encoder(x)
        x = self.ln(x + self.mlp(x[:, 0, :]))
        return x
        
#<-----------Vision Transformer Code Ends Here-----------------------> 


