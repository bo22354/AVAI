import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, scale_factor=8, mode='train', patch_size=48, epoch_size=1000, sample_q=2304, noise=0):
        """
        Args:
            root_dir: Path to 'data' folder
            scale_factor: Downsampling factor (e.g. 4)
            mode: 'train' or 'valid'
            patch_size: Size of the LR crop (Input to Encoder). 
                        Note: This is LR size, not HR size!
            epoch_size: Virtual epoch length
            sample_q: Number of pixels to sample for loss calculation (e.g. 2304)
        """
        self.mode = mode
        self.scale_factor = scale_factor
        self.patch_size = patch_size # Input LR size
        self.epoch_size = epoch_size
        self.sample_q = sample_q
        self.noise = noise
        

        self.read_scale = self.scale_factor
        if self.scale_factor == 16:
            self.read_scale = 8

        # 1. Path Setup (Same as before)
        if mode == 'train':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_train_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_train_LR_bicubic/X{self.read_scale}')
        else:
            self.hr_dir = os.path.join(root_dir, 'DIV2K_valid_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_valid_LR_bicubic/X{self.read_scale}')
            
        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))
        
        if len(self.hr_files) == 0:
            raise RuntimeError(f"No images found in {self.hr_dir}")

    def __len__(self):
        if self.mode == 'train':
            return self.epoch_size
        else:
            return len(self.hr_files)

    def _make_coord(self, shape, flatten=True):
        """ Generates (x,y) coordinates normalized between -1 and 1 """
        coord_seqs = []
        for i, n in enumerate(shape):
            # Generate range -1 to 1
            v0, v1 = -1, 1
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        
        # Stack into grid
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        ret = ret.flip(-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def __getitem__(self, idx):
        # 1. Wrap index for virtual epochs
        file_idx = idx % len(self.hr_files)
        
        # 2. Load Images
        hr_path = self.hr_files[file_idx]
        hr_img = Image.open(hr_path).convert("RGB")
        
        file_name = os.path.basename(hr_path)
        img_id = file_name.split('.')[0]
        lr_path = os.path.join(self.lr_dir, f"{img_id}x{self.read_scale}.png")
        lr_img = Image.open(lr_path).convert("RGB")

        # 3. Crop Logic (Train Only)
        if self.mode == 'train':
            # We crop based on LR size (Encoder Input)
            lr_crop_size = self.patch_size
            hr_crop_size = self.patch_size * self.scale_factor
            
            w, h = lr_img.size
            if w < lr_crop_size or h < lr_crop_size:
                 # Safety resize if image is too small
                 lr_img = lr_img.resize((max(w, lr_crop_size), max(h, lr_crop_size)), Image.BICUBIC)
                 hr_img = hr_img.resize((max(w, lr_crop_size)*self.scale_factor, max(h, lr_crop_size)*self.scale_factor), Image.BICUBIC)
                 w, h = lr_img.size

            i = random.randint(0, h - lr_crop_size)
            j = random.randint(0, w - lr_crop_size)
            
            # Crop LR
            lr_patch = TF.crop(lr_img, i, j, lr_crop_size, lr_crop_size)
            # Crop HR (Multiplied coordinates)
            hr_patch = TF.crop(hr_img, i*self.scale_factor, j*self.scale_factor, hr_crop_size, hr_crop_size)
            
            # Augmentation
            if random.random() < 0.5:
                lr_patch = TF.hflip(lr_patch)
                hr_patch = TF.hflip(hr_patch)
            if random.random() < 0.5:
                lr_patch = TF.rotate(lr_patch, 90)
                hr_patch = TF.rotate(hr_patch, 90)
                
            # Convert to Tensor
            lr_tensor = TF.to_tensor(lr_patch) # [3, 48, 48]
            hr_tensor = TF.to_tensor(hr_patch) # [3, 192, 192]

            # --- TASK 2.3: ON-THE-FLY x16 DOWNSAMPLING ---
            if self.scale_factor == 16:
                # We assume the loaded file is x8. We need to go down one more step (x2).
                # interpolate expects 4D input [Batch, C, H, W], so we unsqueeze(0)
                lr_tensor = torch.nn.functional.interpolate(
                    lr_tensor.unsqueeze(0), 
                    scale_factor=0.5, 
                    mode='bicubic', 
                    align_corners=False
                ).squeeze(0)
                
                # Note: We enforce clamp because bicubic can overshoot 0.0/1.0
                lr_tensor = torch.clamp(lr_tensor, 0.0, 1.0)
            # ---------------------------------------------

            if self.noise > 0:
                noise = torch.randn_like(lr_tensor) * (self.noise / 255.0)
                lr_tensor = lr_tensor + noise
                lr_tensor = torch.clamp(lr_tensor, 0.0, 1.0)
            
            # Normalization (LIIF typically uses standard 0-1 or -1 to 1)
            # Let's use [-1, 1] to match your GAN logic
            lr_tensor = TF.normalize(lr_tensor, [0.5]*3, [0.5]*3)
            hr_tensor = TF.normalize(hr_tensor, [0.5]*3, [0.5]*3)
            
            # 4. LIIF Specific: Coordinate & Pixel Sampling
            
            # Generate grid of coords for the HR patch
            # Shape: [Total_Pixels, 2]
            coords = self._make_coord(hr_tensor.shape[-2:])
            
            # Flatten HR image to match coords [Total_Pixels, 3]
            hr_flat = hr_tensor.permute(1, 2, 0).view(-1, 3)
            
            if self.sample_q is not None:
                # Randomly sample specific pixels to train on
                # This saves memory/time vs training on all 192*192 pixels
                n_pixels = hr_flat.shape[0]
                sample_indices = torch.randperm(n_pixels)[:self.sample_q]
                
                batch_coords = coords[sample_indices]
                batch_gt = hr_flat[sample_indices]
            else:
                batch_coords = coords
                batch_gt = hr_flat

            return {
                'lr': lr_tensor,      # Input for Encoder
                'coord': batch_coords,# (x,y) locations
                'gt': batch_gt        # Target RGB
            }

        else:
            # Validation Mode: Return full images
            lr_tensor = TF.to_tensor(lr_img)
            hr_tensor = TF.to_tensor(hr_img)

            # --- TASK 2.3: ON-THE-FLY x16 DOWNSAMPLING ---
            if self.scale_factor == 16:
                # We assume the loaded file is x8. We need to go down one more step (x2).
                # interpolate expects 4D input [Batch, C, H, W], so we unsqueeze(0)
                lr_tensor = torch.nn.functional.interpolate(
                    lr_tensor.unsqueeze(0), 
                    scale_factor=0.5, 
                    mode='bicubic', 
                    align_corners=False
                ).squeeze(0)
                
                # Note: We enforce clamp because bicubic can overshoot 0.0/1.0
                lr_tensor = torch.clamp(lr_tensor, 0.0, 1.0)
            # ---------------------------------------------

            if self.noise > 0:
                noise = torch.randn_like(lr_tensor) * (self.noise / 255.0)
                lr_tensor = lr_tensor + noise
                lr_tensor = torch.clamp(lr_tensor, 0.0, 1.0)

            # Normalize
            lr_tensor = TF.normalize(lr_tensor, [0.5]*3, [0.5]*3)
            hr_tensor = TF.normalize(hr_tensor, [0.5]*3, [0.5]*3)
            
            # Return standard pair for validation loop
            # (Validation usually infers whole image, so we handle coords in the eval loop)
            return lr_tensor, hr_tensor