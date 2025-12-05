import torch
import argparse
from torch.utils.data import DataLoader
from dataLoader import DIV2KDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pathlib import Path
import torch.nn.functional as F
from torchvision.utils import save_image



# Setup Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

parser = argparse.ArgumentParser(
description="Evaluate Bicubic",
formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path(__file__).parent.parent.parent.resolve()
default_dataset_dir = default_dataset_dir / "data"
parser.add_argument(
    "--dataset-root",
    default=default_dataset_dir,
    help="The location of the dataset to be trained on")
parser.add_argument(
    "--scale-factor",
    default = 8,
    type=int,
    choices=[8, 16],
    help="The scale for which iamges are upscaled to"
)
parser.add_argument(
    "--model",
    default = "best_generator.pth",
    type=str,
    help="path to the model that we want to test the validation data on"
)
parser.add_argument(
    "--noise", 
    default=0, 
    type=float, 
    help="Sigma value for Gaussian noise (e.g. 10, 30, 50)"
)


def main(args):
    print(f"--- Evaluating Bicubic ---")
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/Train")
    validDatasetPath = Path(datasetRoot+"/Valid")
    save_dir = Path("evaluation_results_bicubic")
    save_dir.mkdir(exist_ok=True)
    
    # Load Dataset (Validation Mode)
    val_dataset = DIV2KDataset(
        root_dir=validDatasetPath, 
        scale_factor=args.scale_factor, 
        mode='valid',
        patch_size=0, 
        noise=args.noise
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False
    )

    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for i, (lr, hr) in enumerate(valid_loader):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            # Upscale LR to match HR size using Bicubic interpolation.
            sr = F.interpolate(
                lr, 
                size=hr.shape[2:], 
                mode='nearest', 
            )
            
            # Clamp to valid range [-1, 1] (since interpolation can overshoot)
            sr = torch.clamp(sr, -1, 1)

            # Denormalize [-1, 1] -> [0, 1] for metrics
            sr_norm = torch.clamp(sr * 0.5 + 0.5, 0, 1)
            hr_norm = torch.clamp(hr * 0.5 + 0.5, 0, 1)
            
            lr_norm = torch.clamp(lr * 0.5 + 0.5, 0, 1)

            # Calculate Metrics
            mse = torch.mean((sr_norm - hr_norm) ** 2)
            psnr = -10 * torch.log10(mse).item()
            ssim = ssim_calc(sr_norm, hr_norm).item()

            total_psnr += psnr
            total_ssim += ssim
            count += 1
            
            if i < 10:
                lr_nearest = F.interpolate(lr, size=hr.shape[2:], mode='nearest')
                lr_nearest = torch.clamp(lr_nearest * 0.5 + 0.5, 0, 1)
                
                comparison = torch.cat((lr_nearest, sr_norm, hr_norm), dim=3)
                
                save_path = save_dir / f"val_{i}_psnr{psnr:.2f}.png"
                save_image(comparison, save_path)

            if count % 10 == 0:
                print(f"Processed {count}/{100} images...")

    # Final Results
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    print("\n" + "="*30)
    print(f"RESULTS for Scale x{args.scale_factor}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*30 + "\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)