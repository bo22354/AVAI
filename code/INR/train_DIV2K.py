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
from Trainer import Trainer
from LIIF import LIIF
# from alert import send_loud_notification



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


parser = argparse.ArgumentParser(
    description="Train LIIF (Implicit Neural Representation) on DIV2K",
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
    choices=[2, 3, 4, 8],
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
    "--patch-size",
    default=48,
    type=int,
    help="Size of the LR Image patch size taken per step"
)
parser.add_argument(
    "--sample-q",
    default=2304,
    type=int,
    help="Number of pixels sampled for each querry to each image"
)


def main(args):

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/Train")
    validDatasetPath = Path(datasetRoot+"/Valid")


    print(f"--- Training LIIF | Scale: x{args.scale_factor} ---")
    print(f"LR Patch Size: {args.patch_size} | Sample Q: {args.sample_q}")

    train_dataset = DIV2KDataset(
        root_dir=trainDatasetPath,
        scale_factor=args.scale_factor,
        mode="train",
        patch_size=args.patch_size,
        epoch_size=1000,
        sample_q=args.sample_q
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
        patch_size=args.patch_size,
        epoch_size=1000,
        sample_q=args.sample_q
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=1,
        pin_memory=True,
        num_workers=1
    )

    print(f"Training Images: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Validation Images: {len(valid_dataset)}")

    model = LIIF(n_feats=64, mlp_dim=256).to(DEVICE)

    # 3. Initialize Trainer
    trainer = Trainer(
        model=model,
        device=DEVICE,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scale_factor=args.scale_factor
    )

    # 4. Start Training
    trainer.train(epochs=args.epochs)









if __name__ == "__main__":
    args = parser.parse_args()
    
    # Check path exists before running
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset not found at {args.dataset_root}")
    else:
        main(args)