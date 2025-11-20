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

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
from PIL import Image

import argparse
from pathlib import Path
from multiprocessing import cpu_count

from dataloader import DIV2KDataset
from Generator import Generator
from Discriminator import Discriminator




class Trainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        lr: float = 1e-4,
    ):       
        self.device = device
        self.train_loader = train_loader
        self.netG = generator.to(device)
        self.netD = discriminator.to(device)
        self.optimG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion_GAN = nn.BCEWithLogitsLoss().to(device) 
        self.criterion = nn.L1Loss().to(device)

    def train(
        self,
        epochs: int
    ):
        print("Starting Training...")``
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Metrics for this epoch
            running_results = {'d_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
            self.netG.train()
            self.netD.train()
    
            # 4. The Loop: Note we unpack (lr, hr) from the loader
            for lr_imgs, hr_imgs in self.train_loader:
                batch_size = lr_imgs.size(0)
                
                # Move data to GPU
                lr = lr_imgs.to(self.device)
                hr = hr_imgs.to(self.device)
                
                ############################
                # (1) Update Discriminator
                ############################
                self.optimD.zero_grad()
                
                # Generate Fake HR images
                fake_hr = self.netG(lr)
                
                # Train on Real Images
                # Label 1 = Real (often smoothed to 0.9 for stability)
                pred_real = self.netD(hr)
                label_real = torch.ones_like(pred_real) 
                loss_d_real = self.criterion_GAN(pred_real, label_real)
                
                # Train on Fake Images
                # Label 0 = Fake
                # Detach() is CRITICAL here. We don't want to update Generator weights yet.
                pred_fake = self.netD(fake_hr.detach()) 
                label_fake = torch.zeros_like(pred_fake)
                loss_d_fake = self.criterion_GAN(pred_fake, label_fake)
                
                # Total Discriminator Loss
                loss_d = (loss_d_real + loss_d_fake) / 2
                loss_d.backward()
                self.optimD.step()

                ############################
                # (2) Update Generator
                ############################
                self.optimG.zero_grad()
                
                # We want the Discriminator to think our fakes are Real (Label = 1)
                pred_fake_for_G = self.netD(fake_hr) # No detach here!
                
                # A. Adversarial Loss (The lie)
                loss_g_gan = self.criterion_GAN(pred_fake_for_G, label_real)
                
                # B. Content Loss (The truth) - Are pixels close to Ground Truth?
                loss_g_content = self.criterion_content(fake_hr, hr)
                
                # COMBINED LOSS
                # Standard ratio: 1.0 Content Loss + 0.001 Adversarial Loss
                # If you don't use Content Loss, the GAN hallucinates random images.
                loss_g = loss_g_content + (1e-3 * loss_g_gan)
                
                loss_g.backward()
                self.optimG.step()

                # Tracking
                running_results['g_loss'] += loss_g.item()
                running_results['d_loss'] += loss_d.item()

            # End of Epoch Logging
            avg_d_loss = running_results['d_loss'] / len(self.train_loader)
            avg_g_loss = running_results['g_loss'] / len(self.train_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}] Loss G: {avg_g_loss:.4f} Loss D: {avg_d_loss:.4f} Time: {time.time() - start_time:.2f}s")
 


    def discriminator_loss(self, real_output, fake_output):
        # Loss for real images
        real_loss = self.loss_func(real_output, torch.ones_like(real_output).to(self.device))
        # Loss for fake images
        fake_loss = self.loss_func(fake_output, torch.zeros_like(fake_output).to(self.device))

        loss_D = real_loss + fake_loss
        return loss_D

    def generator_loss(self, fake_output):
        # Compare discriminator's output on fake images with target labels of 1
        loss_G = self.loss_func(fake_output, torch.ones_like(fake_output).to(self.device))
        return loss_G



    def trainDis(x):
        real_x = x.to(self.device)
        real_output = gan_D(real_x)
        fake_x = gan_G(torch.randn([batch_size, input_dim]).to(device)).detach()
        fake_output = gan_D(fake_x)
        loss_D = discriminator_loss(real_output, fake_output)

        # Backpropagate the discriminator loss and update its parameters
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()
        return loss_D
         
    def trainGen(x):
        fake_x = gan_G(torch.randn([batch_size, input_dim]).to(device))
        fake_output = gan_D(fake_x)
        loss_G = generator_loss(fake_output)

        # Backpropagate the generator loss and update its parameters
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()    
        return loss_G   




        