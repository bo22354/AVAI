# import os
# import time

# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from torchvision.utils import save_image

# from PIL import Image


# class Generator(nn.Module):
#     def __init__(self, input_dim):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 32 * 32)
#         self.br1 = nn.Sequential(
#             nn.BatchNorm1d(1024),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
#         self.br2 = nn.Sequential(
#             nn.BatchNorm1d(128 * 7 * 7),
#             nn.ReLU()
#         )
#         self.conv1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),  # Final upsampling to 28x28x1
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.br1(self.fc1(x))
#         x = self.br2(self.fc2(x))
#         # Reshape the tensor for the convolutional layers
#         x = x.reshape(-1, 128, 7, 7)
#         x = self.conv1(x)
#         output = self.conv2(x)
#         return output
    



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

        # 1. First Convolution: Maps Image (3 channels) to Feature Space (64 channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # 2. Residual Blocks (The "Body"): 
        # 16 blocks is standard for SRGAN. You can reduce to 6-8 for faster training.
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(6)])

        # 3. Post-Residual Block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # 4. Upsampling (PixelShuffle)
        # This loop creates the upscaling layers (e.g. two x2 layers for x4 scaling)
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

        # 5. Final Output: Back to 3 Channels (RGB)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        # x is the Low-Res Image [Batch, 3, H, W]
        block1 = self.block1(x)
        res_blocks = self.res_blocks(block1)
        block2 = self.block2(res_blocks)
        
        # Skip connection from start to after res-blocks
        x = self.block3(block1 + block2)
        
        x = self.conv4(x)
        
        # SRGAN usually outputs raw values, but Tanh is common if normalizing to [-1, 1]
        return torch.tanh(x)