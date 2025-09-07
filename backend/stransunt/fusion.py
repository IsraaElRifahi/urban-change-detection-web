import torch
import torch.nn as nn

class CEAFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(CEAFusionModule, self).__init__()

        # Feature Cross Enhancement (FCE)
        self.fce_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.fce_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # FA: Shared convolution → BN → ReLU for each stream
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_f2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Aggregation conv after mul + max
        self.agg_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

        # Final fusion: concat with feat1 & feat2 → BN → ReLU
        self.final_fuse = nn.Sequential(
            nn.BatchNorm2d(in_channels * 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        # ---------- FCE ----------
        att1 = self.sigmoid(self.fce_conv1(feat1))  # attention from feat1
        att2 = self.sigmoid(self.fce_conv2(feat2))  # attention from feat2

        feat1_enh = feat2 * att1 + feat1  # cross-enhancement
        feat2_enh = feat1 * att2 + feat2

        # ---------- FA ----------
        f1 = self.conv_f1(feat1_enh)
        f2 = self.conv_f2(feat2_enh)

        mul = f1 * f2
        max_ = torch.max(f1, f2)

        agg = torch.cat([mul, max_], dim=1)           # [B, 2C, H, W]
        agg_out = self.agg_conv(agg)                  # [B, C, H, W]

        fused = torch.cat([agg_out, feat1, feat2], dim=1)  # [B, 3C, H, W]
        out = self.final_fuse(fused)                       # [B, 3C, H, W]

        return out
