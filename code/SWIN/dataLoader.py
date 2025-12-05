import os
import random
import glob
import torch

from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, scale_factor=8, mode='train', patch_size=96, epoch_size=None, noise = 0):
        """
        root_dir: Path to 'DIV2K' folder
        scale_factor: 8
        mode: 'train' or 'valid'
        patch_size: Size of the HR crop (must be divisible by scale_factor)
        epoch_size: Total number of samples to see per epoch
        noise: The sigma used for the guassian noise added
        """
        self.mode = mode
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.epoch_size = epoch_size
        self.noise = noise
        

        # Instead of creating and storing a x16 bicubic dataset 
        # Use the pre-existing x8 and the downsample "on-the-fly" to save space
        self.read_scale = self.scale_factor
        if self.scale_factor == 16:
            self.read_scale = 8

        if mode == 'train':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_train_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_train_LR_bicubic/X{self.read_scale}')
        elif mode == 'valid':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_valid_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_valid_LR_bicubic/X{self.read_scale}')
            
        
        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*.png"))) # List of HR images, sorted to ensure alignment

        if len(self.hr_files) == 0:
            print(f"\n[ERROR] No images found!")
            print(f"Looking in: {self.hr_dir}")
            print(f"Make sure your folder structure matches this path exactly.\n")
            raise RuntimeError("Dataset is empty.")

    def __len__(self):
        return self.epoch_size if self.mode == 'train' else len(self.hr_files)

    def __getitem__(self, idx):
        file_idx = idx % len(self.hr_files) # If more samples than images wrap back around

        # Load HR Image
        hr_path = self.hr_files[file_idx]
        hr_img = Image.open(hr_path).convert("RGB")
        
        # Create LR Filename 
        file_name = os.path.basename(hr_path)
        img_id = file_name.split('.')[0] # "0001"
        lr_name = f"{img_id}x{self.read_scale}.png"
        lr_path = os.path.join(self.lr_dir, lr_name)
        lr_img = Image.open(lr_path).convert("RGB")

        # Synchronized Random Cropping (Train Only)
        if self.mode == 'train':
            # Determine valid crop coordinates on LR image
            lr_crop_size = self.patch_size // self.read_scale
            
            # Generate random top-left (i, j) for LR
            w, h = lr_img.size
            i = random.randint(0, h - lr_crop_size)
            j = random.randint(0, w - lr_crop_size)
            
            # Crop LR
            lr_patch = TF.crop(lr_img, i, j, lr_crop_size, lr_crop_size)
            
            # Calculate matching coordinates for HR
            # Multiply coordinates by scale factor
            hr_i = i * self.read_scale
            hr_j = j * self.read_scale
            hr_patch = TF.crop(hr_img, hr_i, hr_j, self.patch_size, self.patch_size)
            
            # Random Horizontal Flip (Applied to both)
            if random.random() > 0.5:
                lr_patch = TF.hflip(lr_patch)
                hr_patch = TF.hflip(hr_patch)
                
            # Random Rotation (Applied to both)
            if random.random() > 0.5:
                lr_patch = TF.rotate(lr_patch, 90)
                hr_patch = TF.rotate(hr_patch, 90)
                
        else:
            # Validation: No cropping, pass full images
            lr_patch = lr_img
            hr_patch = hr_img

            

        # Convert to Tensor and Normalize
        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)

        # ON-THE-FLY x16 DOWNSAMPLING
        if self.scale_factor == 16:
            # Assume the loaded file is x8. so need to go down one more step (x2).
            # interpolate expects 4D input [Batch, C, H, W], so we unsqueeze(0)
            lr_tensor = torch.nn.functional.interpolate(
                lr_tensor.unsqueeze(0), 
                scale_factor=0.5, 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
            
            # Clamp because bicubic can overshoot 0.0/1.0
            lr_tensor = torch.clamp(lr_tensor, 0.0, 1.0)

        if self.noise > 0:
            noise = torch.randn_like(lr_tensor) * (self.noise / 255.0)
            lr_tensor = lr_tensor + noise
            lr_tensor = torch.clamp(lr_tensor, 0.0, 1.0)
        
        lr_tensor = TF.normalize(lr_tensor, [0.5]*3, [0.5]*3)
        hr_tensor = TF.normalize(hr_tensor, [0.5]*3, [0.5]*3)

        return lr_tensor, hr_tensor