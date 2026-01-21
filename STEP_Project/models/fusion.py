import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusion, self).__init__()
        
        # Temporal Branch Gates
        self.w_t_f = nn.Linear(feature_dim, feature_dim)
        self.w_t_g = nn.Linear(feature_dim, feature_dim)
        
        # Spatial Branch Gates
        self.w_s_f = nn.Linear(feature_dim, feature_dim)
        self.w_s_g = nn.Linear(feature_dim, feature_dim)

    def forward(self, F_t, F_s):
        # F_t: Temporal Features
        # F_s: Spatial Features
        
        # Eq. 15
        hat_F_t = self.w_t_f(F_t)
        G_t = torch.sigmoid(self.w_t_g(F_t))
        
        hat_F_s = self.w_s_f(F_s)
        G_s = torch.sigmoid(self.w_s_g(F_s))
        
        # Eq. 16: Normalization and Fusion
        G_cat = torch.stack([G_t, G_s], dim=-1)
        G_cat = F.softmax(G_cat, dim=-1) # Normalize weights
        
        # Summation
        F_fusion = hat_F_t * G_cat[..., 0] + hat_F_s * G_cat[..., 1]
        return F_fusion