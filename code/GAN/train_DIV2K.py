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
from Generator import Generator



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


parser = argparse.ArgumentParser(
    description="Train a Siamese Progression Net on HD_EPIC",
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
    default =  4,
    help="The scale for which iamges are upscaled to"
)
parser.add_argument(
    "--batch-size",
    default = 32,
    help="Batch size for forward pass"
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)





def main(args):

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/Train")
    validDatasetPath = Path(datasetRoot+"/Valid")


    patchSize = 24 * args.scale_factor

    train_dataset = DIV2KDataset(
        root_dir=trainDatasetPath,
        scale_factor=args.scale_factor,
        mode="train",
        patch_size=patchSize
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
        patch_size=patchSize
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    print("Number of Training Images: ", len(train_dataset))
    print(f"batches per epoch: {len(train_loader)}")
    print("Number of Validation Images: ", len(valid_dataset))

    
if __name__ == "__main__":
    main(parser.parse_args())






