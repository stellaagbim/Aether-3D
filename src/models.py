import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for 3D Point Clouds.
    
    Unlike a standard Autoencoder, this model learns a probability distribution 
    (mean and variance) for the latent space. This allows for:
    1. Generative sampling (dreaming new shapes).
    2. Smooth interpolation between objects.
    
    Architecture:
        - Encoder: PointNet-based backbone -> predicts mu (mean) and log_var (variance).
        - Reparameterization: z = mu + sigma * epsilon (Gaussian noise).
        - Decoder: Latent vector z -> 3D Point Cloud.
    """

    def __init__(self, latent_dim: int = 128, num_points: int = 1024):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points

        # --- ENCODER (Probabilistic) ---
        # Maps geometric input to a probability distribution
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # The "Variational" Heads
        # We project the 1024 features into TWO vectors: Mean and Log-Variance
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)

        # --- DECODER (Reconstruction) ---
        # Maps a latent sample back to 3D geometry
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, num_points * 3)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The Reparameterization Trick.
        Allows backpropagation through random sampling by shifting the randomness 
        to an auxiliary variable epsilon.
        
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from Unit Gaussian N(0, 1)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input into distribution parameters.
        x: [Batch, 3, N]
        Returns: (mu, log_var)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global Max Pooling (Symmetric Function for Permutation Invariance)
        x = torch.max(x, 2, keepdim=False)[0]
        
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vector z into a Point Cloud.
        """
        batch_size = z.size(0)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to [Batch, 3, N] for Chamfer Loss compatibility
        x = x.view(batch_size, self.num_points, 3)
        return x.transpose(2, 1) # Output: [Batch, 3, N]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main Forward Pass.
        Returns: 
            recon_x: Reconstructed point cloud
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar