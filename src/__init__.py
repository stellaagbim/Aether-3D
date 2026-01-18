"""
Aether-3D: Unsupervised Latent Manifold Learning
Exposing core components for streamlined access.
"""

from .models import VAE
from .dataset import ShapeNetCore
from .loss import AetherLoss
from .dream import AetherDreamer

# This defines what happens when someone types: "from src import *"
__all__ = [
    'VAE',
    'ShapeNetCore', 
    'AetherLoss',
    'AetherDreamer'
]