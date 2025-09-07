import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(CNNEncoder, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = self.conv_block(in_channels, 64)     # Output: B×64×128×128
        self.encoder2 = self.conv_block(64, 256)              # Output: B×256×64×64
        self.encoder3 = self.conv_block(256, 512)             # Output: B×512×32×32

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.encoder1(x)     # B×64×256×256 → pool → 128×128
        p1 = self.pool(f1)        # Bx64x128×128

        f2 = self.encoder2(p1)    # B×256×128×128
        p2 = self.pool(f2)        # B×256×64×64

        f3 = self.encoder3(p2)    # B×512×32×32
        p3 = self.pool(f3) 
        return p1, p2, p3         # Correct channel and size for CEAF and transformer
