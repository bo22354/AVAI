import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Basic Building Blocks ---
class ResBlock(nn.Module):
    """
    Standard Residual Block for the Encoder.
    Structure: Conv -> ReLU -> Conv -> Sum
    """
    def __init__(self, n_feats):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.body(x)

# --- 2. The LIIF Model ---
class LIIF(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=8, mlp_dim=256):
        super(LIIF, self).__init__()

        # A. THE ENCODER (CNN)
        # This acts like a standard SR network (EDSR/RDN) but WITHOUT the upsampling at the end.
        # It maps the LR image to a "Feature Grid" of the same spatial size.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            *[ResBlock(n_feats) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        # B. THE DECODER (MLP)
        # This is the "Implicit" part. It takes a feature vector and a coordinate
        # and predicts the RGB value.
        # Input Dimension: n_feats (from Encoder) + 2 (x,y coordinate)
        self.decoder = nn.Sequential(
            nn.Linear(n_feats + 2, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 3) # Output is RGB (3 channels)
        )

    def query_features(self, feature_map, coords):
        """
        Retrieves the feature vector from the feature_map that is spatially 
        closest to the query coordinates.
        
        feature_map: [Batch, Channels, H, W]
        coords:      [Batch, N_Queries, 2]  (Normalized -1 to 1)
        """
        
        # F.grid_sample expects coordinates in shape [Batch, H_out, W_out, 2]
        # We treat our list of N queries as a "1 pixel high, N pixels wide" image
        # to trick grid_sample into sampling them for us.
        
        # 1. Reshape coords: [B, N, 2] -> [B, 1, N, 2]
        coords = coords.unsqueeze(1)
        
        # 2. Sample
        # mode='nearest': We want the EXACT feature vector of the nearest center, not an average.
        # align_corners=False: Standard for modern PyTorch grids.
        samples = F.grid_sample(
            feature_map, 
            coords, 
            align_corners=False, 
            padding_mode='border',
            mode='nearest' 
        )
        # samples shape is now: [Batch, Channels, 1, N]
        
        # 3. Reshape back: [Batch, Channels, 1, N] -> [Batch, N, Channels]
        return samples.view(samples.shape[0], samples.shape[1], -1).permute(0, 2, 1)

    def forward(self, lr_image, coords):
        """
        lr_image: [Batch, 3, H, W]
        coords:   [Batch, N_Queries, 2]
        """
        # 1. Encode the LR Image
        # Output: [Batch, 64, H, W]
        features = self.encoder(lr_image)

        # 2. Query Features
        # For every (x,y) in coords, find the nearest feature vector in 'features'
        # Output: [Batch, N, 64]
        feat_vectors = self.query_features(features, coords)

        # 3. Prepare Input for MLP
        # Concatenate the Feature Vector with the Coordinate itself
        # This tells the MLP: "I am at this location (coords) and here is the context (feat)"
        # Input: [Batch, N, 64+2]
        inp = torch.cat([feat_vectors, coords], dim=-1)

        # 4. Predict RGB
        # Output: [Batch, N, 3]
        rgb_pred = self.decoder(inp)
        
        return rgb_pred