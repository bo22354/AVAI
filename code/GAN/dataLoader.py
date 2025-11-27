import os
import random
import glob
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, scale_factor=8, mode='train', patch_size=96, epoch_size=None):
        """
        root_dir: Path to 'DIV2K' folder
        scale_factor: 4
        mode: 'train' or 'test'
        patch_size: Size of the HR crop (must be divisible by scale_factor)
        """
        self.mode = mode
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.epoch_size = epoch_size
        
        # 1. Define paths based on DIV2K structure
        # Structure: DIV2K/DIV2K_train_HR/0001.png
        #            DIV2K/DIV2K_train_LR_bicubic/X4/0001x4.png
        if mode == 'train':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_train_HR')
            print("HR Path: ", self.hr_dir)
            self.lr_dir = os.path.join(root_dir, f'DIV2K_train_LR_bicubic/X{scale_factor}')
        elif mode == 'valid':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_valid_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_valid_LR_bicubic/X{scale_factor}')
            
        # Get list of HR images, sorted to ensure alignment
        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))

        if len(self.hr_files) == 0:
            print(f"\n[ERROR] No images found!")
            print(f"Looking in: {self.hr_dir}")
            print(f"Make sure your folder structure matches this path exactly.\n")
            raise RuntimeError("Dataset is empty.")

    def __len__(self):
        return self.epoch_size if self.mode == 'train' else len(self.hr_files)

    def __getitem__(self, idx):
        # if epoch_size > actual files then will need to wrap back around to the start
        file_idx = idx % len(self.hr_files)

        # 1. Load HR Image
        hr_path = self.hr_files[file_idx]
        hr_img = Image.open(hr_path).convert("RGB")
        
        # 2. Construct LR Filename matching DIV2K convention
        # HR: 0001.png -> LR: 0001x4.png
        file_name = os.path.basename(hr_path)
        img_id = file_name.split('.')[0] # "0001"
        lr_name = f"{img_id}x{self.scale_factor}.png"
        lr_path = os.path.join(self.lr_dir, lr_name)
        
        lr_img = Image.open(lr_path).convert("RGB")

        # 3. Synchronized Random Cropping (Train Only)
        if self.mode == 'train':
            # A. Determine valid crop coordinates on LR image
            lr_crop_size = self.patch_size // self.scale_factor
            
            # Generate random top-left (i, j) for LR
            # Use standard Python random, not PyTorch transforms
            w, h = lr_img.size
            i = random.randint(0, h - lr_crop_size)
            j = random.randint(0, w - lr_crop_size)
            
            # B. Crop LR
            lr_patch = TF.crop(lr_img, i, j, lr_crop_size, lr_crop_size)
            
            # C. Calculate matching coordinates for HR
            # Multiply coordinates by scale factor
            hr_i = i * self.scale_factor
            hr_j = j * self.scale_factor
            hr_patch = TF.crop(hr_img, hr_i, hr_j, self.patch_size, self.patch_size)
            
            # D. Random Horizontal Flip (Applied to both)
            if random.random() > 0.5:
                lr_patch = TF.hflip(lr_patch)
                hr_patch = TF.hflip(hr_patch)
                
            # E. Random Rotation (Applied to both)
            if random.random() > 0.5:
                lr_patch = TF.rotate(lr_patch, 90)
                hr_patch = TF.rotate(hr_patch, 90)
                
        else:
            # Validation: No cropping, pass full images
            lr_patch = lr_img
            hr_patch = hr_img

        # 4. Convert to Tensor and Normalize
        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)
        
        # Optional: Normalize to [-1, 1] for GANs
        # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] logic
        lr_tensor = TF.normalize(lr_tensor, [0.5]*3, [0.5]*3)
        hr_tensor = TF.normalize(hr_tensor, [0.5]*3, [0.5]*3)

        return lr_tensor, hr_tensor