import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Decoder stages: [input channels + skip channels] → output channels
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = DecoderBlock(512, 512)  # f5 + f4 → 512

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DecoderBlock(512 + 3*512, 256)  # + f3

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DecoderBlock(256 + 3*256, 128)  # + f2

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DecoderBlock(128 + 3*64, 64)    # + f1

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)  # 2-class prediction
        )

    def forward(self, f4, f3, f2, f1):
      x = self.up3(f4)                        # H/16 → H/8
      x = torch.cat([x, f3], dim=1)
      x = self.dec3(x)

      x = self.up2(x)                         # H/8 → H/4
      x = torch.cat([x, f2], dim=1)
      x = self.dec2(x)

      x = self.up1(x)                         # H/4 → H/2
      x = torch.cat([x, f1], dim=1)
      x = self.dec1(x)

      x = self.up0(x)                         # H/2 → H
      out = self.final_conv(x)

      return out  # shape: [B, 2, H, W]
