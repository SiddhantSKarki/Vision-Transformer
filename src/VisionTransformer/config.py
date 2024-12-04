import torch

'''
    ViT Config File
'''
class ViTConfig:
    def __init__(self,
                 input_channels,
                 num_classes,
                 num_patches,
                 embedding_size,
                 patch_size,
                 num_heads,
                 num_blocks,
                 device,
                 batch_size,
                 dropout=0.2,
                 precision=torch.float32):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.precision = precision
        self.device = device
        self.batch_size = batch_size