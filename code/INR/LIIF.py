import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Basic Building Blocks ---
class EDSRBlock(nn.Module):
    """
    Enhanced Deep Residual Block.
    Differences from Standard ResBlock:
    1. No Batch Normalization (Better for Super-Resolution).
    2. Constant Scaling (res_scale) to stabilize training.
    """
    def __init__(self, n_feats, res_scale=1.0):
        super(EDSRBlock, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return x + res

# --- 2. The LIIF Model ---
class LIIF(nn.Module):

    def __init__(self, n_feats=64, n_resblocks=16, mlp_dim=256):
        """
        n_resblocks: Increased default from 8 to 16 for better quality.
        """
        super(LIIF, self).__init__()

        # A. THE ENCODER (EDSR Style)
        # 1. Head (Extract shallow features)
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)

        # 2. Body (Deep feature extraction)
        # We stack 16 EDSR blocks. 
        # We use res_scale=0.1 to prevent exploding gradients in deep nets.
        self.body = nn.Sequential(
            *[EDSRBlock(n_feats, res_scale=0.1) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        # B. THE DECODER (MLP) - Unchanged
        self.decoder = nn.Sequential(
            # nn.Linear(n_feats + 2, mlp_dim),
            nn.Linear(n_feats + 6, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 3), # Output RGB
            nn.Tanh()              # Range [-1, 1]
        )

    def query_features(self, feature_map, coords):
        """
        Calculates Features AND Relative Coordinates.
        """
        # feature_map: [B, C, H, W]
        # coords: [B, N, 2] in range [-1, 1]
        
        # 1. Prepare grid_sample
        # We assume coords are (x, y) thanks to your .flip(-1) fix earlier
        bs, n_coords, _ = coords.shape
        h_feat, w_feat = feature_map.shape[-2:]
        
        # 2. Retrieve the nearest features (Nearest Neighbor lookup)
        # [B, C, N]
        samples = F.grid_sample(
            feature_map, 
            coords.unsqueeze(1), 
            align_corners=False, 
            padding_mode='border',
            mode='bilinear' 
        )
        samples = samples.view(bs, -1, n_coords).permute(0, 2, 1) # [B, N, C]
        
        # 3. Calculate Relative Coordinates (THE UPGRADE)
        # We need to convert continuous range [-1, 1] into grid indices [0, H-1]
        
        # Transform coords from [-1, 1] to [0, H-1] space
        coord_grid_x = (coords[:, :, 0] + 1) / 2 * (w_feat - 1)
        coord_grid_y = (coords[:, :, 1] + 1) / 2 * (h_feat - 1)
        
        # Round to find the "center" of the nearest feature pixel
        # .round() gives us the integer index of the nearest feature
        grid_x_center = torch.round(coord_grid_x)
        grid_y_center = torch.round(coord_grid_y)
        
        # Calculate the distance from the query to that center
        # This tells the MLP: "I am 0.2 units away from this feature"
        rel_x = coord_grid_x - grid_x_center
        rel_y = coord_grid_y - grid_y_center
        
        # Normalize relative coords back to roughly [-1, 1] range for stability
        # Each cell is 1 unit wide in grid space, so the max distance is 0.5.
        # Multiplying by 2 ensures the MLP sees inputs roughly in [-1, 1]
        rel_coord = torch.stack([rel_x, rel_y], dim=-1) * 2
        
        return samples, rel_coord



    def forward(self, lr_image, coords):
        # 1. Encode with Skip Connection
        # Standard EDSR logic: Head -> Body -> Add Head (Global Residual)
        x = self.head(lr_image)
        res = self.body(x)
        features = x + res  # Global Skip connection

        # 2. Query
        feat_vectors, rel_coords = self.query_features(features, coords)

        # --- NEW: Positional Encoding ---
        # We boost the 2D coords into 6D by adding sin/cos features
        # This helps the MLP "see" high frequency changes
        rel_fourier = torch.cat([
            rel_coords,
            torch.sin(rel_coords * 3.1415),
            torch.cos(rel_coords * 3.1415)
        ], dim=-1)
        # -------------------------------

        # 3. Concatenate (Note: Decoder input size increases by +4)
        inp = torch.cat([feat_vectors, rel_fourier], dim=-1)
        
        # 4. Predict
        rgb_pred = self.decoder(inp)
        
        return rgb_pred