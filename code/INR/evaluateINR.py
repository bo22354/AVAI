import torch
import argparse
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import torch.nn.functional as F

# --- Imports (Local) ---
# Ensure we can find the sibling files LIIF.py and dataset_liif.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataLoader import DIV2KDataset
from LIIF import LIIF

# --- Setup Device ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- Arguments ---
parser = argparse.ArgumentParser(description="Evaluate INR (LIIF) Model")

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
    choices=[2, 3, 4, 8], 
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
def make_coord(shape):
    """ Helper to generate (x,y) coordinates for the full image grid """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    
    # Stack and Flip to match (x, y) order for grid_sample
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    ret = ret.flip(-1) 
    return ret.view(-1, ret.shape[-1])

def batched_predict(model, lr, coord, chunk_size=30000):
    """ Predicts pixels in chunks to avoid CUDA OOM on large images """
    preds = []
    n_pixels = coord.shape[1]
    
    for i in range(0, n_pixels, chunk_size):
        coord_chunk = coord[:, i:i+chunk_size, :]
        pred_chunk = model(lr, coord_chunk)
        preds.append(pred_chunk)
    
    return torch.cat(preds, dim=1)

def evaluate(args):
    print(f"--- Evaluating INR Model: {args.model} ---")
    print(f"--- Scale: x{args.scale_factor} ---")

    save_dir = Path("evaluation_results")
    save_dir.mkdir(exist_ok=True)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/Train")
    validDatasetPath = Path(datasetRoot+"/Valid")

    # 1. Load Dataset (Validation Mode)
    val_dataset = DIV2KDataset(
        root_dir=validDatasetPath,
        scale_factor=args.scale_factor,
        mode="valid",
        patch_size=48, # Ignored in valid
        epoch_size=1,  # Ignored in valid
        sample_q=None  # None = Full Image
    )

    valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. Initialize Model
    # Ensure these params match your training config!
    # If you used the Advanced LIIF (EDSR), keep resblocks=16. 
    # If you used the Simple LIIF, change back to 8.
    model = LIIF(n_feats=64, n_resblocks=16, mlp_dim=256).to(DEVICE)

    # 3. Load Weights
    try:
        state_dict = torch.load(args.model, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("-> Weights loaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        return

    model.eval()

    # 4. Metrics
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    total_psnr = 0
    total_ssim = 0
    count = 0

    print(f"Starting inference on {len(val_dataset)} images...")

    with torch.no_grad():
        for i, (lr, hr) in enumerate(valid_loader):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            
            # A. Generate Query Coords for the Full HR Image
            h_hr, w_hr = hr.shape[-2:]
            coord = make_coord((h_hr, w_hr)).to(DEVICE).unsqueeze(0)
            
            # B. Predict (Batched)
            pred_all = batched_predict(model, lr, coord)
            
            # C. Reshape to Image [1, 3, H, W]
            sr = pred_all.view(1, h_hr, w_hr, 3).permute(0, 3, 1, 2)

            # D. Denormalize & Clamp
            # LIIF output (Tanh) is [-1, 1] -> [0, 1]
            sr_norm = torch.clamp(sr * 0.5 + 0.5, 0, 1)
            hr_norm = torch.clamp(hr * 0.5 + 0.5, 0, 1)

            # E. Calculate Metrics
            mse = torch.mean((sr_norm - hr_norm) ** 2)
            psnr = -10 * torch.log10(mse).item()
            ssim = ssim_calc(sr_norm, hr_norm).item()

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            # --- VISUALIZATION LOGIC ---
            if args.save_images and i < 10: # Only save the first 10 images
                # Resize LR to match HR size for side-by-side comparison
                # We use 'nearest' so you can clearly see the blocky pixels of the input
                lr_resized = F.interpolate(lr, size=(h_hr, w_hr), mode='nearest')
                lr_resized = torch.clamp(lr_resized * 0.5 + 0.5, 0, 1)
                
                # Stack: Left=Input, Middle=Generated, Right=Ground Truth
                comparison = torch.cat((lr_resized, sr_norm, hr_norm), dim=3)
                save_path = save_dir / f"val_{i}_psnr{psnr:.2f}.png"
                save_image(comparison, save_path)
                print(f"Saved visualization: {save_path}")
            # ---------------------------
            
            if count % 10 == 0:
                print(f"Processed {count}/{len(val_dataset)}... (Current Avg PSNR: {total_psnr/count:.2f})")

    # 5. Final Results
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