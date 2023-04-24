import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size,in_chans,patch_size, embed_size=768):
        super().__init__()
        self. img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size//patch_size) **2 #number of patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_size,
            kernel_size = patch_size,
            stride = patch_size
        )

    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2) # flatten last two dimensions
        x= x.transpose(1,2) # transpose the dims
        return x

class Attention(nn.Module):
    pass

class MLP(nn.Module):
    pass

class VisionTransformer(nn.Module):
    pass