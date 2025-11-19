#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path
import sys
from dataloader import ProgressionDataset 
import os

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a Siamese Progression Net on HD_EPIC",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path(os.getcwd() + "/dataset")
parser.add_argument(
    "--dropout", 
    default=0.5, 
    type=float,
    help="The dropout rate in the final fully connected layer")
parser.add_argument(
    "--sgd-momentum", 
    default=0, 
    type=float)
parser.add_argument(
    "--dataset-root",
    default=default_dataset_dir,
    help="The location of the dataset to be trained on")
parser.add_argument(
    "--log-dir", 
    default=Path("logs"), 
    type=Path,
    help="The path to the folder containing training logs for tensorboard")
parser.add_argument(
    "--learning-rate",
    default=1e-2,
    type=float,
    help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=16,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

classifications = ["before", "after", "different"]

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transformList = []
    transformList.append(transforms.ToTensor())
    transform = transforms.Compose(transformList)
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    # TODO put in the correct train and test datasets
    datasetRoot = str(args.dataset_root)
    path = Path(datasetRoot+"/train")
    folder_names = [item.name for item in path.iterdir() if item.is_dir()]
    
    train_dataset = ProgressionDataset(
        root_dir=datasetRoot+"/train", 
        mode="train", 
        transform = transform,
        recipe_ids_list=folder_names,
        epoch_size=args.epochs
        )

    test_dataset = ProgressionDataset(
        root_dir=datasetRoot+"/test",
        mode="test",
        transform = transform,
        epoch_size=args.epochs,
        label_file=datasetRoot+"/test_labels.txt"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = SPN(height=112, width=112, channels=3, class_count=3, dropout=args.dropout)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum = args.sgd_momentum)
    # TODO set parameters correctly
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, scheduler
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class SPN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout:float):
        super().__init__()
        # Create network structure
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        # Convolutional Section
        # First Layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1norm2 = nn.BatchNorm2d(64)

        # Second Layer
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=128,
            kernel_size=(3,3),
            padding=(1,1),
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))
        self.conv2norm2 = nn.BatchNorm2d(128)
        
        # Third Layer
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=256,
            kernel_size=(3,3),
            padding=(1,1),
        )
        self.initialise_layer(self.conv3)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))
        self.conv3norm2 = nn.BatchNorm2d(256)

        # Forth Layer
        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=512,
            kernel_size=(3,3),
            padding=(1,1),
        )
        self.initialise_layer(self.conv4)
        self.pool4 = nn.AdaptiveAvgPool2d((1,1))
        self.conv4norm2 = nn.BatchNorm2d(512)

        # Fully Connected Section
        self.fc1 = nn.Linear(1024, 512)
        self.initialise_layer(self.fc1)
        self.norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 3)
        self.initialise_layer(self.fc2)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(3)

    def forward(self, anchor: torch.Tensor, comparator: torch.Tensor) -> torch.Tensor:
        # First layer pass
        anc = F.relu(self.conv1norm2(self.conv1(anchor)))
        anc = self.pool1(anc)
        comp = F.relu(self.conv1norm2(self.conv1(comparator)))
        comp = self.pool1(comp)
        
        # Second layer pass
        anc = F.relu(self.conv2norm2(self.conv2(anc)))
        anc = self.pool2(anc)
        comp = F.relu(self.conv2norm2(self.conv2(comp)))
        comp = self.pool2(comp)

        # Third layer pass
        anc = F.relu(self.conv3norm2(self.conv3(anc)))
        anc = self.pool3(anc)
        comp = F.relu(self.conv3norm2(self.conv3(comp)))
        comp = self.pool3(comp)

        # Forth layer pass
        anc = F.relu(self.conv4norm2(self.conv4(anc)))
        anc = self.pool4(anc)
        comp = F.relu(self.conv4norm2(self.conv4(comp)))
        comp = self.pool4(comp)

        # Concatinate layers
        anc = torch.flatten(input=anc, start_dim=1)
        comp = torch.flatten(input=comp, start_dim=1)
        x = torch.cat((anc, comp), dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.fc2(x)))
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        scheduler,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch_anc, batch_comp, labels in self.train_loader:
                batch_anc = batch_anc.to(self.device)
                batch_comp = batch_comp.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                ## Compute the forward pass of the model
                logits = self.model.forward(batch_anc, batch_comp)

                ## Compute the loss 
                loss = self.criterion(logits, labels)

                ## Compute the backward pass
                loss.backward()

                ## Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                # self.validate() will put the model in validation mode
                self.validate()
                # switch back to train mode afterwards
                self.model.train()

            self.scheduler.step()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch_anc, batch_comp, labels in self.val_loader:
                batch_anc = batch_anc.to(self.device)
                batch_comp = batch_comp.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch_anc, batch_comp)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        classAcc = compute_per_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        for c in range(len(classifications)):
            print(f"{classifications[c]} accuracy: {classAcc[c] * 100:2.2f}")





def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute the overall accuracy for the dataset


    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def compute_per_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> [float]:
    """
    Compute the accuracy for each class within the dataset
    
    
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)

    acc = np.array([0 for c in classifications], dtype=float)
    unique, counts = np.unique(labels, return_counts=True)
    denoms = dict(zip(unique, counts))

    for c in range(len(classifications)):
        acc[c] = float(((labels == preds) * (preds == c)).sum()) / denoms[c]
    return acc





def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"dropout={args.dropout}_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"momentum={args.sgd_momentum}_" +
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())