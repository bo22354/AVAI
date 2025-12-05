import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Basic Building Blocks ---
class EDSRBlock(nn.Module):
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

class LIIF(nn.Module):

    def __init__(self, n_feats=64, n_resblocks=16, mlp_dim=256):
        super(LIIF, self).__init__()

        ##  THE ENCODER (EDSR Style)
        # Head (Extract shallow features)
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)

        # Body (Deep feature extraction)
        self.body = nn.Sequential(
            *[EDSRBlock(n_feats, res_scale=0.1) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        ## THE DECODER (MLP)
        self.decoder = nn.Sequential(
            nn.Linear(n_feats + 6, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 3), 
            nn.Tanh()              
        )

    def query_features(self, feature_map, coords):
        # Calculates Features AND Relative Coordinates.

        
        # Prepare grid_sample
        bs, n_coords, _ = coords.shape
        h_feat, w_feat = feature_map.shape[-2:]
        
        # Retrieve the nearest features (Nearest Neighbor lookup)
        samples = F.grid_sample(
            feature_map, 
            coords.unsqueeze(1), 
            align_corners=False, 
            padding_mode='border',
            mode='bilinear' 
        )
        samples = samples.view(bs, -1, n_coords).permute(0, 2, 1) # [B, N, C]
        
        # Calculate Relative Coordinates
        coord_grid_x = (coords[:, :, 0] + 1) / 2 * (w_feat - 1)
        coord_grid_y = (coords[:, :, 1] + 1) / 2 * (h_feat - 1)
        
        grid_x_center = torch.round(coord_grid_x)
        grid_y_center = torch.round(coord_grid_y)
        
        rel_x = coord_grid_x - grid_x_center
        rel_y = coord_grid_y - grid_y_center
        
        # Normalize relative coords back to roughly [-1, 1] range for stability
        rel_coord = torch.stack([rel_x, rel_y], dim=-1) * 2
        
        return samples, rel_coord



    def forward(self, lr_image, coords):
        # Encode with Skip Connection
        x = self.head(lr_image)
        res = self.body(x)
        features = x + res  

        # Query
        feat_vectors, rel_coords = self.query_features(features, coords)

        # Positional Encoding 
        # 2D coords into 6D by adding sin/cos features, helps the MLP "see" high frequency changes
        rel_fourier = torch.cat([
            rel_coords,
            torch.sin(rel_coords * 3.1415),
            torch.cos(rel_coords * 3.1415)
        ], dim=-1)

        # Concatenate
        inp = torch.cat([feat_vectors, rel_fourier], dim=-1)
        
        # Predict
        rgb_pred = self.decoder(inp)
        return rgb_pred