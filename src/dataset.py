import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from typing import Optional

class ShapeNetCore(Dataset):
    """
    Dataset Wrapper for Unsupervised Geometric Learning.
    
    This pipeline:
    1. Loads the ModelNet10/40 dataset.
    2. Discards classification labels (Unsupervised).
    3. Normalizes shapes to the unit sphere (Critical for VAE stability).
    4. Samples exactly 1024 points per shape.
    """

    def __init__(self, root: str = 'data/ModelNet10', train: bool = True, num_points: int = 1024):
        self.num_points = num_points
        
        # The Transformation Pipeline
        # We normalize scale to [-1, 1] to help the VAE converge
        self.transform = Compose([
            SamplePoints(num_points, remove_faces=True, include_normals=False),
            NormalizeScale()
        ])
        
        # Load Raw Data using PyG's efficient caching system
        print(f">> [DataEngine] Initializing {'Train' if train else 'Test'} Split...")
        self.dataset = ModelNet(
            root=root,
            name='10',
            train=train,
            transform=self.transform
        )
        print(f">> [DataEngine] Loaded {len(self.dataset)} shapes.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            points: Tensor of shape [3, 1024]
        """
        data = self.dataset[idx]
        
        # Extract only position data (geometry)
        pos = data.pos  # [N, 3]
        
        # Transpose to [3, N] because Conv1d expects channels first
        pos = pos.transpose(0, 1)
        
        return pos.float()