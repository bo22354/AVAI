import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual  # The "Skip Connection"

class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        n_resblocks = 16
        if scale_factor == 16:
            n_resblocks = 24

        # First Convolution: Maps Image (3 channels) to Feature Space (64 channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Residual Blocks (The "Body"): 
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(n_resblocks)])

        # Post-Residual Block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling (PixelShuffle)
        block3 = [
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        ]
        for _ in range(upsample_block_num - 1):
            block3.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            block3.append(nn.PixelShuffle(2))
            block3.append(nn.PReLU())
        self.block3 = nn.Sequential(*block3)

        # Final Output: Back to 3 Channels (RGB)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        block1 = self.block1(x)
        res_blocks = self.res_blocks(block1)
        block2 = self.block2(res_blocks)
        
        x = self.block3(block1 + block2)
        x = self.conv4(x)
        
        return torch.tanh(x)