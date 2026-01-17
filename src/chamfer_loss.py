import torch
import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        """
        Calculates the Chamfer Distance between two point clouds x and y.
        x: [Batch, N, 3] (Generated)
        y: [Batch, N, 3] (Ground Truth)
        """
        # 1. Compute pairwise distances
        # Expand x and y to allow broadcasting
        x_expand = x.unsqueeze(2) # [B, N, 1, 3]
        y_expand = y.unsqueeze(1) # [B, 1, N, 3]
        
        # Squared distance matrix [B, N, N]
        dist = torch.sum((x_expand - y_expand) ** 2, dim=3)
        
        # 2. For every point in X, find the nearest point in Y
        min_dist_x, _ = torch.min(dist, dim=2)
        
        # 3. For every point in Y, find the nearest point in X
        min_dist_y, _ = torch.min(dist, dim=1)
        
        # 4. Average the distances
        loss = torch.mean(min_dist_x) + torch.mean(min_dist_y)
        
        return loss