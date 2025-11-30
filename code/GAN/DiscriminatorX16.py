import torch
import torch.nn as nn

class DiscriminatorX16(nn.Module):
    def __init__(self, input_shape=(3, 96, 96)):
        super(DiscriminatorX16, self).__init__()

        channels, height, width = input_shape

        # Helper block: Conv -> BatchNorm -> LeakyReLU
        def discriminator_block(in_filters, out_filters, stride=1, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, stride=1, normalize=False), # Block 1: 3 -> 64 channels. No BN
            *discriminator_block(64, 64, stride=2), # Block 2: Downsample to 96x96
            *discriminator_block(64, 128, stride=1), # Block 3: 64 -> 128 channels
            *discriminator_block(128, 128, stride=2), # Block 4: Downsample to 48x48
            *discriminator_block(128, 256, stride=1), # Block 5: 128 -> 256 channels
            *discriminator_block(256, 256, stride=2), # Block 6: Downsample to 24x24
            *discriminator_block(256, 512, stride=1), # Block 7: 256 -> 512 channels
            *discriminator_block(512, 512, stride=2), # Block 8: Downsample to 12x12

            # Extra Layers for x16 scale_factor
            *discriminator_block(512, 1024, stride=1), # Block 9: 512 -> 1024 channels
            *discriminator_block(1024, 1024, stride=2), # Block 10: Downsample to 6x6
        )

        # Classification Head
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6)) # Adaptive Pooling used incase I decide to try bigger patch size e.g. 32x32
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 6 * 6, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1),
            nn.Sigmoid() 
        )

    def forward(self, img):
        features = self.model(img) # Convolutions
        features = self.avg_pool(features) # Pooling
        features_flat = torch.flatten(features, 1) # Flattern
        validity = self.classifier(features_flat) # Classifier        
        return validity