import os
import torch
import time
import sys

import numpy as np
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Utils.calculatePSNR import calculate_psnr


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        scale_factor: int,
        noise: int,
        lr: float = 2e-4, # SwinIR typically uses 2e-4
    ):       
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.to(device)
        self.scale_factor = scale_factor
        
        # Optimizer: Adam or AdamW is standard for Transformers
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Scheduler: MultiStepLR is standard for SwinIR
        # Decays at 50% and 80% of total epochs usually
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[250, 400, 450, 475], gamma=0.5
        )
        
        # Loss: L1 Loss (Charbonnier Loss is better, but L1 is standard enough)
        self.criterion = nn.L1Loss().to(device)
        self.noise = noise

    def train(
        self, 
        epochs: int
    ):
        print("Starting SwinIR Training...")
        best_psnr = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            running_loss = 0.0

            self.model.train()
    
            for i, (lr, hr) in enumerate(self.train_loader):
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward Pass
                sr = self.model(lr)
                
                # Loss
                loss = self.criterion(sr, hr)
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Update Scheduler
            self.scheduler.step()
            
            # Logging
            avg_loss = running_loss / len(self.train_loader)
            current_lr = self.scheduler.get_last_lr()[0]

            save_dir = "Models/noise"+str(self.noise)+"/scale"+str(self.scale_factor)+"/"
            os.makedirs(save_dir,exist_ok=True)
            torch.save(self.model.state_dict(), save_dir+"last_SWIN.pth")
            
            # Validate every 5 epochs
            if epoch % 5 == 0:
            # if epoch % 5 == 0 and epoch > 75: #TODO replace once know works
                avg_psnr = self.validate()
                print(f"Val PSNR: {avg_psnr:.2f} dB")                
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    strPSNR = f"{best_psnr:.2f}"
                    torch.save(self.model.state_dict(), f"{save_dir}/{strPSNR}_SWIN.pth")
                
                # Save Visualization
                self.visualize(epoch)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.5f} | Time: {time.time() - start_time:.1f}s")

    def validate(self):
        self.model.eval()
        total_psnr = 0.0
        
        with torch.no_grad():
            for lr, hr in self.valid_loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                sr = self.model(lr)
                
                # Crop to match dimensions (SwinIR sometimes outputs fixed window sizes)
                _, _, h_sr, w_sr = sr.shape
                _, _, h_hr, w_hr = hr.shape
                h_min, w_min = min(h_sr, h_hr), min(w_sr, w_hr)
                
                sr = sr[:, :, :h_min, :w_min]
                hr = hr[:, :, :h_min, :w_min]
                
                # Clamp outputs (SwinIR doesn't have Tanh, so it can overshoot)
                # But typically trained on normalized data, so outputs are roughly correct.
                # Assuming data is [-1, 1], we clamp to [-1, 1]
                sr = torch.clamp(sr, -1, 1)
                
                total_psnr += calculate_psnr(sr, hr).item()
                
        return total_psnr / len(self.valid_loader)

    def visualize(self, epoch):
        self.model.eval()
        with torch.no_grad():
            lr, hr = next(iter(self.valid_loader))
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            sr = self.model(lr)
            
            # Resize LR for comparison
            lr_resized = torch.nn.functional.interpolate(lr, size=hr.shape[2:], mode='nearest')
            
            # Crop SR to match HR
            _, _, h, w = hr.shape
            sr = sr[:, :, :h, :w]
            
            comparison = torch.cat((lr_resized, sr, hr), dim=3)
            save_dir = "./results/noise"+str(self.noise)+"/scale"+str(self.scale_factor)
            os.makedirs(save_dir, exist_ok=True)
            save_image(comparison * 0.5 + 0.5, f"{save_dir}/epoch_{epoch}.png")