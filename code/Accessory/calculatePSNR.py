import math
import torch

def calculate_psnr(img1, img2):
    # img1 and img2 are tensors in range [-1, 1]
    # 1. Denormalize to [0, 1] range
    img1 = img1 * 0.5 + 0.5
    img2 = img2 * 0.5 + 0.5
    
    # 2. Calculate MSE
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    
    # 3. Calculate PSNR
    # 20 * log10(MAX / sqrt(MSE))
    # Since range is [0, 1], MAX is 1.
    return 20 * torch.log10(1.0 / torch.sqrt(mse))