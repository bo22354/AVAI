import numpy as np
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
from PIL import Image
import tensorflow_datasets as tfds
import argparse
from pathlib import Path
from dataloader import DIV2KDataset


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


parser = argparse.ArgumentParser(
    description="Train a Siamese Progression Net on HD_EPIC",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path(os.getcwd() + "/dataset")
parser.add_argument(
    "--dataset-root",
    default=default_dataset_dir,
    help="The location of the dataset to be trained on")






def main(args):

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    trainDatasetPath = Path(datasetRoot+"/train")


    # dataset = tfds.load('div2k/bicubic_x4', split='train', shuffle_files=True) # Potential way to use dataset

    train_dataset = DIV2KDataset(
        root_dir=trainDatasetPath,
        scale_factor=args.scale_factor,
        mode="train",
        patch_size=
    )

    train_loader = torch.utils.data.DataLoader(

    )