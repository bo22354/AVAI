import torch
import argparse
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)

from dataLoader import DIV2KDataset
from SwinIR import SwinIR

# Setup Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

parser = argparse.ArgumentParser(
    description="Evaluate SwinIR Model"
)
default_dataset_dir = Path(__file__).parent.parent.parent.resolve() / "data"

parser.add_argument(
    "--dataset-root", 
    default=default_dataset_dir, 
    help="Path to data folder"
    )
parser.add_argument(
    "--scale-factor", 
    default=8, 
    type=int, 
    choices=[8, 16], 
    help="Upscaling scale"
    )
parser.add_argument(
    "--model", 
    required=True, 
    type=str, 
    help="Path to the .pth model file"
    )
parser.add_argument(
    "--save-images",
    action="store_true",
    default = True,
    help="Save comparison images"
)
parser.add_argument(
    "--noise", 
    default=0, 
    type=float, 
    help="Sigma value for Gaussian noise (e.g. 10, 30, 50)"
)

def evaluate(args):
    print(f"--- Evaluating SwinIR Model: {args.model} ---")
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/Train")
    validDatasetPath = Path(datasetRoot+"/Valid")
    save_dir = Path("evaluation_results_swin")
    save_dir.mkdir(exist_ok=True)

    lr_patchSize = 48 

    # Load Dataset
    val_dataset = DIV2KDataset(
        root_dir=validDatasetPath,
        scale_factor=args.scale_factor,
        mode="valid",
        patch_size=0, 
        noise=args.noise 
    )
    valid_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False
    )

    # Configure Model
    model = SwinIR(
        img_size=lr_patchSize,
        patch_size=1, 
        in_chans=3,
        embed_dim=60,
        depths=[6, 6, 6, 6], 
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2, 
        upscale=args.scale_factor,
        img_range=1., 
        resi_connection='1conv'
    )
    try:
        state_dict = torch.load(args.model, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("-> Weights loaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        return

    model.to(DEVICE)
    model.eval()

    # Metrics
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    total_psnr = 0
    total_ssim = 0
    count = 0

    print(f"Starting inference on {len(val_dataset)} images...")

    with torch.no_grad():
        for i, (lr, hr) in enumerate(valid_loader):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            sr = model(lr)

            # Crop to min common size (handle padding artifacts)
            _, _, h_sr, w_sr = sr.shape
            _, _, h_hr, w_hr = hr.shape
            h_min = min(h_sr, h_hr)
            w_min = min(w_sr, w_hr)
            
            sr = sr[:, :, :h_min, :w_min]
            hr = hr[:, :, :h_min, :w_min]

            # Denormalize & Clamp
            sr_norm = torch.clamp(sr * 0.5 + 0.5, 0, 1)
            hr_norm = torch.clamp(hr * 0.5 + 0.5, 0, 1)

            # Metrics
            mse = torch.mean((sr_norm - hr_norm) ** 2)
            psnr = -10 * torch.log10(mse).item()
            ssim = ssim_calc(sr_norm, hr_norm).item()

            total_psnr += psnr
            total_ssim += ssim
            count += 1
            
            if args.save_images and i < 10:
                # Resize LR to match HR size
                lr_resized = F.interpolate(lr, size=(h_min, w_min), mode='nearest')
                lr_resized = torch.clamp(lr_resized * 0.5 + 0.5, 0, 1)
                
                comparison = torch.cat((lr_resized, sr_norm, hr_norm), dim=3)
                save_path = save_dir / f"val_{i}_psnr{psnr:.2f}.png"
                save_image(comparison, save_path)
            
            if count % 10 == 0:
                print(f"Processed {count}/{len(val_dataset)}...")

    # Final Results
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print("\n" + "="*40)
    print(f"FINAL RESULTS (Scale x{args.scale_factor})")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root not found at {args.dataset_root}")
    else:
        evaluate(args)