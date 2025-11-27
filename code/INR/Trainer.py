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
from Accessory.calculatePSNR import calculate_psnr

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        scale_factor: int,
        lr: float = 1e-4,
    ):       
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.to(device)
        
        # LIIF uses Adam, usually with a learning rate decay (simplified here)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # L1 Loss is standard for LIIF (Pixel-wise accuracy)
        self.criterion = nn.L1Loss().to(device)
        self.scale_factor = scale_factor

    def train(self, epochs: int):
        print("Starting LIIF Training...")
        best_psnr = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Metrics
            running_loss = 0.0
            self.model.train()
    
            # 1. The Loop
            # Note: batch is now a DICTIONARY {'lr', 'coord', 'gt'}
            for batch in self.train_loader:
                
                # Move data to GPU
                lr = batch['lr'].to(self.device)
                coord = batch['coord'].to(self.device) # (x,y) coordinates
                gt = batch['gt'].to(self.device)       # Ground Truth RGB
                
                ############################
                # Update LIIF Model
                ############################
                self.optimizer.zero_grad()
                
                # Forward Pass: Model takes Image AND Coordinates
                # Output shape: [Batch, Sample_Q, 3] (List of pixels)
                pred = self.model(lr, coord)
                
                # Calculate Loss (Compare predicted RGB to Ground Truth RGB)
                loss = self.criterion(pred, gt)
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # End of Epoch Logging
            avg_loss = running_loss / len(self.train_loader)
            
            # 2. Validate
            avg_psnr = self.validate()
            
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.5f} | Val PSNR: {avg_psnr:.2f} dB | Time: {time.time() - start_time:.2f}s")

            # 3. Save Model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                strPSNR = f"{best_psnr:.2f}"
                os.makedirs("Models_LIIF", exist_ok=True)
                torch.save(self.model.state_dict(), f"Models_LIIF/{strPSNR}_best_liif.pth")
                print("-> New Best Model Saved!")

            os.makedirs("Models_LIIF", exist_ok=True)
            torch.save(self.model.state_dict(), "Models_LIIF/last_liif.pth")

            # 4. Visualization
            if epoch % 5 == 0:
                self.visualize(epoch)

    def _make_coord(self, shape, ranges=None):
        """ Helper to generate full image coordinates for Validation/Viz """
        coord_seqs = []
        for i, n in enumerate(shape):
            v0, v1 = -1, 1
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        return ret.view(-1, ret.shape[-1])

    def validate(self):
        """
        Validation loop is tricky for LIIF because we need to query 
        EVERY pixel to reconstruct the image for PSNR calculation.
        """
        self.model.eval()
        total_psnr = 0.0
        
        with torch.no_grad():
            for lr, hr in self.valid_loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                # 1. Generate Coords for the FULL HR Image
                # hr.shape is [1, 3, H, W]
                h_hr, w_hr = hr.shape[-2:]
                coord = self._make_coord((h_hr, w_hr)).to(self.device)
                
                # Add batch dim: [1, H*W, 2]
                coord = coord.unsqueeze(0) 
                
                # 2. Inference (Full Image)
                # If you get CUDA OOM here, you need to query in chunks (batched inference)
                # But for DIV2K validation crops, this usually fits.
                pred_rgb = self.model(lr, coord)
                
                # 3. Reshape Prediction back to Image
                # [1, H*W, 3] -> [1, H, W, 3] -> [1, 3, H, W]
                sr = pred_rgb.view(1, h_hr, w_hr, 3).permute(0, 3, 1, 2)
                
                # 4. Calculate PSNR
                # Note: LIIF naturally handles arbitrary sizes, so no cropping needed usually,
                # but good to be safe.
                total_psnr += calculate_psnr(sr, hr).item()
                
        return total_psnr / len(self.valid_loader)

    def visualize(self, epoch):
        """ Reconstructs one image for saving """
        self.model.eval()
        with torch.no_grad():
            # Grab one batch
            lr, hr = next(iter(self.valid_loader))
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            # Generate Coords
            h_hr, w_hr = hr.shape[-2:]
            coord = self._make_coord((h_hr, w_hr)).to(self.device).unsqueeze(0)
            
            # Predict
            pred_rgb = self.model(lr, coord)
            sr = pred_rgb.view(1, h_hr, w_hr, 3).permute(0, 3, 1, 2)
            
            # Resize LR for comparison
            lr_resized = nn.functional.interpolate(lr, size=hr.shape[2:], mode='nearest')
            
            # Concatenate
            comparison = torch.cat((lr_resized, sr, hr), dim=3)
            
            save_dir = "./results_liif"
            os.makedirs(save_dir, exist_ok=True)
            save_image(comparison * 0.5 + 0.5, f"{save_dir}/epoch_{epoch}.png")