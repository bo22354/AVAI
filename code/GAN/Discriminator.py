import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 96, 96)):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Helper block: Conv -> BatchNorm -> LeakyReLU
        def discriminator_block(in_filters, out_filters, stride=1, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Block 1: Input (3) -> 64 channels. No BN on first layer.
            *discriminator_block(channels, 64, stride=1, normalize=False),
            
            # Block 2: Downsample to 48x48
            *discriminator_block(64, 64, stride=2),
            
            # Block 3: 64 -> 128
            *discriminator_block(64, 128, stride=1),
            
            # Block 4: Downsample to 24x24
            *discriminator_block(128, 128, stride=2),
            
            # Block 5: 128 -> 256
            *discriminator_block(128, 256, stride=1),
            
            # Block 6: Downsample to 12x12
            *discriminator_block(256, 256, stride=2),
            
            # Block 7: 256 -> 512
            *discriminator_block(256, 512, stride=1),
            
            # Block 8: Downsample to 6x6
            *discriminator_block(512, 512, stride=2),
        )

        # The Classification Head
        # We use AdaptiveAvgPool so this works even if you change patch size from 96 to 128
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid() # Output probability [0, 1]
        )

    def forward(self, img):
        # 1. Extract Features through Convolutions
        features = self.model(img)
        
        # 2. Force features to 6x6 spatial size
        features = self.avg_pool(features)
        
        # 3. Flatten (Batch_Size, 18432)
        features_flat = torch.flatten(features, 1)
        
        # 4. Classify (Real vs Fake)
        validity = self.classifier(features_flat)
        
        return validity