import torch
import torch.nn as nn

from stransunt.encoder import CNNEncoder            # Only outputs f1, f2, f3
from stransunt.fusion import CEAFusionModule
from stransunt.transformer import TransformerEncoder
from stransunt.decoder import Decoder

class STransUNet(nn.Module):
    def __init__(self):
        super(STransUNet, self).__init__()

        # CNN Encoder: only 3 levels
        self.encoder = CNNEncoder()  # Should output f1, f2, f3 (H/2, H/4, H/8)

        # CEAF modules for levels 1, 2, 3
        self.ceaf1 = CEAFusionModule(in_channels=64)
        self.ceaf2 = CEAFusionModule(in_channels=256)
        self.ceaf3 = CEAFusionModule(in_channels=512)

        # Downsample encoder output (f3) before Transformer
        self.down_to_f4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # H/8 → H/16
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Transformer on downsampled encoder output
        self.transformer = TransformerEncoder(
            in_channels=1024,
            embed_dim=768,
            depth=12,
            num_heads=12,
            img_size=16  # H/16 if input H = 256
        )

        self.reduce_trans = nn.Conv2d(768, 512, kernel_size=1)

        # Conv after CEAF4 fusion to reduce channels before decoder
        self.ceaf4 = CEAFusionModule(in_channels=512)
        self.reduce_ceaf4 = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder: takes f4 (H/16), f3 (H/8), f2 (H/4), f1 (H/2)
        self.decoder = Decoder()

    def forward(self, img_t1, img_t2):
        # ===== Encoder: CNN for 3 levels =====
        f1_t1, f2_t1, f3_t1 = self.encoder(img_t1)  # f3_t1: B×512×32×32
        f1_t2, f2_t2, f3_t2 = self.encoder(img_t2)
        # ===== CEAF1–3 =====
        f1 = self.ceaf1(f1_t1, f1_t2)  # [B, 3x64, H/2]
        f2 = self.ceaf2(f2_t1, f2_t2)  # [B, 3x128, H/4]
        f3 = self.ceaf3(f3_t1, f3_t2)  # [B, 3x256 = 768, H/8] → skip connection only
        # ===== Downsample encoder output f3 → f4 =====
        f4_t1 = self.down_to_f4(f3_t1)  # [B, 1024, 16, 16]
        f4_t2 = self.down_to_f4(f3_t2)
        # ===== Transformer on f4 =====
        f4_1_trans = self.transformer(f4_t1)
        f4_2_trans = self.transformer(f4_t2)

        f4_1 = self.reduce_trans(f4_1_trans)  # [B, 512, 16, 16]
        f4_2 = self.reduce_trans(f4_2_trans)

        # ===== CEAF4 Fusion → Conv to 512 =====
        ceaf4_out = self.ceaf4(f4_1, f4_2)           # [B, 3x512, 16, 16]
        f4 = self.reduce_ceaf4(ceaf4_out)            # [B, 512, 16, 16]

        # ===== Decode =====
        out = self.decoder(f4, f3, f2, f1)           # [B, 2, H, W]
        return out
