import os
import torch
import time
import sys

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
        lr: float = 1e-4,
    ):       
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.to(device)
        self.noise = noise
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[200, 350, 450], 
            gamma=0.5
        )
        self.criterion = nn.L1Loss().to(device)
        self.scale_factor = scale_factor

    def train(self, epochs: int):
        print("Starting LIIF Training")
        best_psnr = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            running_loss = 0.0

            self.model.train()
    
            i = 0
            # bathc is now a DICTIONARY {'lr', 'coord', 'gt'}

            for batch in self.train_loader:
                i += 1

                # Move data to GPU
                lr = batch['lr'].to(self.device)
                coord = batch['coord'].to(self.device) 
                gt = batch['gt'].to(self.device)       
                
                # Update LIIF Model
                self.optimizer.zero_grad()
                
                pred = self.model(lr, coord)
                
                # Calculate Loss
                loss = self.criterion(pred, gt)
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
                        
            save_dir = "Models/noise"+str(self.noise)+"/scale"+str(self.scale_factor)+"/"
            os.makedirs(save_dir,exist_ok=True)
            torch.save(self.model.state_dict(), save_dir+"/last_LIIF.pth")

            # Visualization
            if epoch % 5 == 0 and epoch > 75:
                avg_psnr = self.validate()
                print(f"Val PSNR: {avg_psnr:.2f} dB")
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    strPSNR = f"{best_psnr:.2f}"
                    torch.save(self.model.state_dict(), f"{save_dir}/{strPSNR}_LIIF.pth")
                    print("-> New Best Model Saved!")

                self.visualize(epoch)

            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.5f} | Time: {time.time() - start_time:.2f}s")
            self.scheduler.step()


    def _make_coord(self, shape, ranges=None):
        # Helper to generate full image coordinates for Validation
        coord_seqs = []
        for i, n in enumerate(shape):
            v0, v1 = -1, 1
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        ret = ret.flip(-1)
        return ret.view(-1, ret.shape[-1])

    def validate(self):
        self.model.eval()
        total_psnr = 0.0
        
        with torch.no_grad():
            for lr, hr in self.valid_loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                # Generate Coords for Full Image
                h_hr, w_hr = hr.shape[-2:]
                coord = self._make_coord((h_hr, w_hr)).to(self.device).unsqueeze(0)
                
                # Batched Inference
                pred_rgb = self.batched_predict(lr, coord, chunk_size=30000)
                
                # Reshape and Calculate PSNR
                sr = pred_rgb.view(1, h_hr, w_hr, 3).permute(0, 3, 1, 2)
                
                total_psnr += calculate_psnr(sr, hr).item()
                
        return total_psnr / len(self.valid_loader)

    def visualize(self, epoch):
        self.model.eval()
        with torch.no_grad():
            lr, hr = next(iter(self.valid_loader))
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            h_hr, w_hr = hr.shape[-2:]
            coord = self._make_coord((h_hr, w_hr)).to(self.device).unsqueeze(0)
            
            pred_rgb = self.batched_predict(lr, coord, chunk_size=30000)
            
            sr = pred_rgb.view(1, h_hr, w_hr, 3).permute(0, 3, 1, 2)
            
            # Resize LR for comparison
            lr_resized = nn.functional.interpolate(lr, size=hr.shape[2:], mode='nearest')
            
            comparison = torch.cat((lr_resized, sr, hr), dim=3)
            save_dir = "./results/noise"+str(self.noise)+"/scale"+str(self.scale_factor)
            os.makedirs(save_dir, exist_ok=True)
            save_image(comparison * 0.5 + 0.5, f"{save_dir}/epoch_{epoch}_full_image.png")


    def batched_predict(self, lr, coord, chunk_size=30000):
        # Splits the coordinate grid into chunks and predicts them sequentially to avoid OOM errors on large images.

        self.model.eval()
        prediction_list = []
        n_pixels = coord.shape[1] # Total pixels to predict
        
        with torch.no_grad():
            for i in range(0, n_pixels, chunk_size):
                # Grab a small slice of coordinates
                coord_chunk = coord[:, i:i+chunk_size, :]
                
                # Predict just this slice
                pred_chunk = self.model(lr, coord_chunk)
                
                # Store result (move to CPU if VRAM is extremely tight, else keep on GPU)
                prediction_list.append(pred_chunk)
        
        # Stitch all chunks back together
        return torch.cat(prediction_list, dim=1)