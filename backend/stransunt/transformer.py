import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1024, embed_dim=768, patch_size=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] → [B, N, embed_dim] the format required for Transformers 
        x = self.proj(x)  # [B, embed_dim, H', W'] Projects in_channels=1024 to embed_dim=768
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim], converts [B, 768, 16, 16] → [B, 256, 768]
        return x, (H, W)

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels=1024, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, img_size=16):
        super().__init__()
        #Create patch embeddings and calculate how many patch tokens (sequence length) you'll have
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        self.num_patches = img_size * img_size  # e.g. 16x16 patches

        # Learnable positional embedding :This is added to patch embeddings so the Transformer knows where each patch comes from.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True   # this enables PreNorm (LayerNorm → MSA → Residual)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)        # [B, N, embed_dim]
        
        # Add positional embedding here: positional information to each token
        x = x + self.pos_embed[:, :x.size(1)]   # [B, N, embed_dim]
        
        x = self.transformer(x)
        x = self.norm(x)

        # Reshape back to feature map: output feature map passed to the fusion module (CEAF) and the decoder.
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)  # [ B , 768, 16, 16]  Converts token sequence back into spatial feature map
        return x