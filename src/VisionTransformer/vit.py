import torch
import torch.nn as nn
from . import transformer, config


'''
Vision Transformer Code Starts Here
'''
class PatchEmbedding(nn.Module):
    def __init__(self,
                input_channels,
                embedding_size,
                patch_size,
                batch_size,
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
        x = self.sequence(x).permute(0, 2, 1)
        x = torch.cat([x, self.cls_tkn_embd], dim=1)
        x = x + self.pos_embd
        return x
    


class VisionTransformer(nn.Module):
    def __init__(self,
                config):
        super().__init__()

        self.emdeddings = PatchEmbedding(
                input_channels=config.input_channels,
                embedding_size=config.embedding_size,
                patch_size=config.patch_size,
                num_patches=config.num_patches,
                precision=config.precision,
                batch_size=config.batch_size,
                device=config.device
        )
        
        self.transformer_encoder = transformer.TransformerEncoder(
            embedding_size=config.embedding_size,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            block_size=config.num_patches+1,
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_size, 4*config.embedding_size),
            nn.GELU(),
            nn.LayerNorm(4*config.embedding_size),
            nn.Linear(4*config.embedding_size, config.num_classes)
        )


    def forward(self, x):
        x = self.emdeddings(x)
        x = self.transformer_encoder(x)
        return self.mlp(x[:, 0, :])