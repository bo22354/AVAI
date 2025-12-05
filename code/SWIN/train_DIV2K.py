import os
import torch
import time

import numpy as np

import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.utils import save_image

from PIL import Image

import argparse
from pathlib import Path
from multiprocessing import cpu_count

from dataLoader import DIV2KDataset
from dataLoader import DIV2KDataset 
from SwinIR import SwinIR
from Trainer import Trainer



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


parser = argparse.ArgumentParser(
    description="Train SwinIR on DIV2K",
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
    default =  8,
    type=int,
    choices=[8, 16],
    help="The scale for which iamges are upscaled to"
)
parser.add_argument(
    "--batch-size",
    default = 16,
    type = int,
    help="Batch size for training"
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--epochs",
    default = 500,
    type=int,
    help="Number of epochs that are run for training"
)
parser.add_argument(
    "--noise", 
    default=0, 
    type=float, 
    help="Sigma value for Gaussian noise (e.g. 10, 30, 50)"
)




def main(args):
    #Dataset Path and Folders
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/Train")
    validDatasetPath = Path(datasetRoot+"/Valid")

    lr_patchSize = 32 # Increase to 48 for x8 scaling
    patchSize = lr_patchSize * args.scale_factor 
    print(f"Scale: x{args.scale_factor} | HR Patch Size: {patchSize}")

    # Initialise Datasets from dataloader
    train_dataset = DIV2KDataset(
        root_dir=trainDatasetPath,
        scale_factor=args.scale_factor,
        mode="train",
        patch_size=patchSize,
        epoch_size=1000,
        noise=args.noise
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    valid_dataset = DIV2KDataset(
        root_dir=validDatasetPath,
        scale_factor=args.scale_factor,
        mode="valid",
        patch_size=patchSize,
        epoch_size=1000,
        noise=args.noise
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
    )

    print(f"Training Images: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Validation Images: {len(valid_dataset)}")

    # Model Configuration
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

    trainer = Trainer(
        model=model,
        device=DEVICE,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scale_factor=args.scale_factor,
        noise=args.noise
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset not found at {args.dataset_root}")
    else:
        main(args)