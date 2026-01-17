import torch
import torch.nn as nn

class AetherLoss(nn.Module):
    """
    Composite Loss Function for VAE Training.
    
    Loss = Reconstruction_Loss + (Beta * KL_Divergence)
    
    1. Reconstruction Loss: Chamfer Distance (Geometric similarity).
    2. KL Divergence: Forces latent space to approximate N(0, 1).
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def chamfer_distance(self, x, y):
        # x: [B, 3, N] -> [B, N, 3]
        x = x.transpose(2, 1)
        y = y.transpose(2, 1)
        
        x_expand = x.unsqueeze(2) 
        y_expand = y.unsqueeze(1) 
        
        # Squared Euclidean distance
        dist = torch.sum((x_expand - y_expand) ** 2, dim=3)
        
        min_dist_x, _ = torch.min(dist, dim=2)
        min_dist_y, _ = torch.min(dist, dim=1)
        
        return torch.mean(min_dist_x) + torch.mean(min_dist_y)

    def kl_divergence(self, mu, logvar):
        """
        Analytical KL Divergence for Gaussian Distributions.
        KLD = -0.5 * sum(1 + log(var) - mu^2 - var)
        """
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / mu.size(0) # Normalize by batch size

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = self.chamfer_distance(recon_x, x)
        kld_loss = self.kl_divergence(mu, logvar)
        
        total_loss = recon_loss + (self.beta * kld_loss)
        return total_loss, recon_loss, kld_loss