import torch
import argparse
from torch.utils.data import DataLoader
from dataLoader import DIV2KDataset
from Generator import Generator
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pathlib import Path
from torchvision.utils import save_image
import torch.nn.functional as F


# Setup Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


parser = argparse.ArgumentParser(
    description="Evaluate SRGAN",
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

def main(args):
    print(f"--- Evaluating Model: {args.model} ---")
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/Train")
    validDatasetPath = Path(datasetRoot+"/Valid")
    
    # 1. Load Dataset (Validation Mode)
    # Important: batch_size=1 because images have different sizes
    val_dataset = DIV2KDataset(
        root_dir=validDatasetPath, 
        scale_factor=args.scale_factor, 
        mode='valid',
        patch_size=0, # Not used in valid mode
        noise=args.noise
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False
    )

    model = Generator(scale_factor=args.scale_factor).to(DEVICE)


    # Load weights
    try:
        state_dict = torch.load(args.model, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    evaluate(model, valid_loader, len(val_dataset), DEVICE, args.scale_factor)


def evaluate(model, valid_loader, datasetLength, DEVICE, scale_factor): 
    print(f"--- Evaluating on Validation ---")
    save_dir = Path("evaluation_results")
    save_dir.mkdir(exist_ok=True)


    model.eval()

    # 3. Define Metrics
    # SSIM expects [0, 1] range
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    total_psnr = 0
    total_ssim = 0
    count = 0

    # 4. Inference Loop
    with torch.no_grad():
        for i, (lr, hr) in enumerate(valid_loader):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            # Upscale
            sr = model(lr)

            #Crop both images to be the same size
            _, _, h_sr, w_sr = sr.shape
            _, _, h_hr, w_hr = hr.shape
            
            h_min = min(h_sr, h_hr)
            w_min = min(w_sr, w_hr)
            
            sr = sr[:, :, :h_min, :w_min]
            hr = hr[:, :, :h_min, :w_min]

            # Denormalize from [-1, 1] to [0, 1] for metrics
            sr_norm = sr * 0.5 + 0.5
            hr_norm = hr * 0.5 + 0.5
            
            # Clamp to ensure we don't go slightly beyond 0 or 1 due to float math
            sr_norm = torch.clamp(sr_norm, 0, 1)
            hr_norm = torch.clamp(hr_norm, 0, 1)

            # Calculate PSNR
            mse = torch.mean((sr_norm - hr_norm) ** 2)
            psnr = -10 * torch.log10(mse).item()
            
            # Calculate SSIM
            ssim = ssim_calc(sr_norm, hr_norm).item()

            total_psnr += psnr
            total_ssim += ssim
            count += 1
                        # --- VISUALIZATION LOGIC ---
            if args.save_images and i < 10: # Only save the first 10 images
                # Resize LR to match HR size for side-by-side comparison
                # We use 'nearest' so you can clearly see the blocky pixels of the input
                lr_resized = F.interpolate(lr, size=(h_min, w_min), mode='nearest')
                lr_resized = torch.clamp(lr_resized * 0.5 + 0.5, 0, 1)
                
                # Stack: Left=Input, Middle=Generated, Right=Ground Truth
                comparison = torch.cat((lr_resized, sr_norm, hr_norm), dim=3)
                save_path = save_dir / f"val_{i}_psnr{psnr:.2f}.png"
                save_image(comparison, save_path)
                print(f"Saved visualization: {save_path}")

            # ---------------------------
            if count % 10 == 0:
                print(f"Processed {count}/{datasetLength} images...")

    # 5. Final Results
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    print("\n" + "="*30)
    print(f"RESULTS for Scale x{scale_factor}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*30 + "\n")

    




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)