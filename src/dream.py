import torch
import torch.nn.functional as F
from src.models import VAE
from src.dataset import ShapeNetCore
import plotly.graph_objects as go
import numpy as np
import os

class AetherDreamer:
    """
    The Generative Interface for Aether-3D.
    """
    
    def __init__(self, model_path='checkpoints/aether_best.pth', device='cuda'):
        # Auto-detect CPU if CUDA is missing (Critical for your laptop)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f">> [Dreamer] Loading Aether Brain from {model_path} on {self.device}...")
        
        # Load the trained brain
        self.model = VAE(latent_dim=128).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print(f"!! [Error] Model file {model_path} not found. Did training finish?")
            return
        
        # Load data just to get real shapes to morph between
        self.dataset = ShapeNetCore(train=False)
        
    def _plot_cloud(self, points, title, color_scale='Viridis'):
        """Generates a 3D Plotly Figure."""
        if isinstance(points, torch.Tensor):
            points = points.cpu().detach().numpy()
        
        # Handle shape mismatch: Plotly wants [3, N] or [N, 3] depending on setup
        # Our dataset returns [3, 1024], so we use rows 0,1,2 for x,y,z
        fig = go.Figure(data=[go.Scatter3d(
            x=points[0, :], y=points[1, :], z=points[2, :],
            mode='markers',
            marker=dict(size=3, color=points[1, :], colorscale=color_scale, opacity=0.8)
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    def morph(self, idx1, idx2, steps=10):
        """
        Takes two objects (e.g., Desk and Bathtub),
        extracts their DNA, and generates the mutant shapes in between.
        """
        print(f">> [Dreamer] Morphing between Object #{idx1} and #{idx2}...")
        
        # 1. Get Real Objects from Dataset
        obj1 = self.dataset[idx1].unsqueeze(0).to(self.device) # [1, 3, 1024]
        obj2 = self.dataset[idx2].unsqueeze(0).to(self.device)
        
        # 2. Extract Latent DNA (z1, z2)
        with torch.no_grad():
            mu1, _ = self.model.encode(obj1)
            mu2, _ = self.model.encode(obj2)
            
        # 3. Interpolate (The Dream Path)
        alphas = np.linspace(0, 1, steps)
        
        for i, alpha in enumerate(alphas):
            # Mix the DNA
            z_mix = (1 - alpha) * mu1 + alpha * mu2
            
            # Decode the mixture into a 3D shape
            with torch.no_grad():
                shape_mix = self.model.decode(z_mix)[0] # [3, N]
            
            # Save the frame
            filename = f"morph_step_{i}.html"
            self._plot_cloud(shape_mix, f"Morphing: {int(alpha*100)}%").write_html(filename)
            print(f"   Saved frame: {filename}")
            
        print(">> Morphing complete. Open 'morph_step_5.html' to see the hybrid.")

if __name__ == "__main__":
    # Create the dreamer
    dreamer = AetherDreamer()
    # Try morphing object #10 (Desk) into object #50 (Bathtub)
    # You can change these numbers to try different shapes
    dreamer.morph(idx1=10, idx2=50, steps=10)