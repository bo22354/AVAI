import os
import torch
import time

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from PIL import Image

from calculatePSNR import calculate_psnr
from evaluate import evaluate




class Trainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        scale_factor: int,
        noise: int,
        lr: float = 1e-4,
    ):       
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.netG = generator.to(device)
        self.netD = discriminator.to(device)
        self.optimG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion_GAN = nn.BCEWithLogitsLoss().to(device) 
        self.criterion = nn.L1Loss().to(device)
        self.scale_factor = scale_factor
        self.noise = noise

    def train(
        self,
        epochs: int
    ):
        print("Starting Training...")
        best_psnr = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()            
            # Metrics for this epoch
            running_results = {'d_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
            self.netG.train()
            self.netD.train()
    
            # 4. The Loop: Note we unpack (lr, hr) from the loader
            for lr_imgs, hr_imgs in self.train_loader:
                batch_size = lr_imgs.size(0)
                
                # Move data to GPU
                lr = lr_imgs.to(self.device)
                hr = hr_imgs.to(self.device)
                
                ############################
                # (1) Update Discriminator
                ############################
                self.optimD.zero_grad()
                
                # Generate Fake HR images
                fake_hr = self.netG(lr)
                
                # Train on Real Images
                # Label 1 = Real (often smoothed to 0.9 for stability)
                pred_real = self.netD(hr)
                label_real = torch.ones_like(pred_real) 
                loss_d_real = self.criterion_GAN(pred_real, label_real)
                
                # Train on Fake Images
                # Label 0 = Fake
                # Detach() is CRITICAL here. We don't want to update Generator weights yet.
                pred_fake = self.netD(fake_hr.detach()) 
                label_fake = torch.zeros_like(pred_fake)
                loss_d_fake = self.criterion_GAN(pred_fake, label_fake)
                
                # Total Discriminator Loss
                loss_d = (loss_d_real + loss_d_fake) / 2
                loss_d.backward()
                self.optimD.step()

                ############################
                # (2) Update Generator
                ############################
                self.optimG.zero_grad()
                
                # We want the Discriminator to think our fakes are Real (Label = 1)
                pred_fake_for_G = self.netD(fake_hr) # No detach here!
                
                # A. Adversarial Loss (The lie)
                loss_g_gan = self.criterion_GAN(pred_fake_for_G, label_real)

                
                # B. Content Loss (The truth) - Are pixels close to Ground Truth?
                loss_g_content = self.criterion(fake_hr, hr)
                
                # COMBINED LOSS
                # Standard ratio: 1.0 Content Loss + 0.001 Adversarial Loss
                # If you don't use Content Loss, the GAN hallucinates random images.
                loss_g = loss_g_content + (1e-3 * loss_g_gan)
                
                loss_g.backward()
                self.optimG.step()

                # Tracking
                running_results['g_loss'] += loss_g.item()
                running_results['d_loss'] += loss_d.item()

            # End of Epoch Logging
            avg_d_loss = running_results['d_loss'] / len(self.train_loader)
            avg_g_loss = running_results['g_loss'] / len(self.train_loader)

            save_dir = "Models/noise"+str(self.noise)+"/scale"+str(self.scale_factor)+"/"
            os.makedirs(save_dir,exist_ok=True)
            torch.save(self.netG.state_dict(), save_dir+"last_GAN.pth")

            if epoch % 5 == 0 and epoch > 75:
                avg_psnr = self.validate()
                print(f"Val PSNR: {avg_psnr:.2f} dB")
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    strPSNR = f"{best_psnr:.2f}"
                    # os.makedirs("Models_GAN", exist_ok=True)
                    torch.save(self.netG.state_dict(), f"{save_dir}/{strPSNR}_GAN.pth")
                    print("-> New Best Model Saved!")

                # 1. Resize LR to match HR size for visualization (using Nearest Neighbor or Bilinear)
                lr_resized = nn.functional.interpolate(lr, scale_factor=self.scale_factor, mode='nearest')
                
                # 2. Concatenate: LR (resized) | Generated | Ground Truth
                # dim=3 places them side-by-side horizontally
                comparison = torch.cat((lr_resized, fake_hr, hr), dim=3) 
                
                # 3. Undo Normalization [-1, 1] -> [0, 1] for saving
                save_dir = "./results/noise"+str(self.noise)+"/scale"+str(self.scale_factor)
                os.makedirs(save_dir, exist_ok=True)
                
                # 4. Save
                save_image(comparison * 0.5 + 0.5, f"{save_dir}/epoch_{epoch}.png")
     
            print(f"Epoch [{epoch+1}/{epochs}] Loss G: {avg_g_loss:.4f} Loss D: {avg_d_loss:.4f} Time: {time.time() - start_time:.2f}s")


    def validate(self):
        self.netG.eval()
        total_psnr = 0.0
        
        with torch.no_grad():
            for lr, hr in self.valid_loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                # 1. Inference
                sr = self.netG(lr)
                
                # 2. Dynamic Cropping (The Fix)
                # Get the minimum dimensions between Generated (sr) and Ground Truth (hr)
                batch, c, h_sr, w_sr = sr.shape
                _, _, h_hr, w_hr = hr.shape
                
                h_min = min(h_sr, h_hr)
                w_min = min(w_sr, w_hr)
                
                # Crop both tensors to the smallest common size
                sr_cropped = sr[:, :, :h_min, :w_min]
                hr_cropped = hr[:, :, :h_min, :w_min]
                
                # 3. Calculate Metric on the cropped versions
                total_psnr += calculate_psnr(sr_cropped, hr_cropped).item()
                
        return total_psnr / len(self.valid_loader)


    