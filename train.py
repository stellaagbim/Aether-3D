import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models import VAE
from src.dataset import ShapeNetCore
from src.chamfer_loss import AetherLoss
import os
import time
from tqdm import tqdm  # Professional progress bar

class AetherTrainer:
    """
    Research-Grade VAE Trainer with KL-Annealing.
    
    Features:
    - Automatic Checkpointing (Saves best model).
    - KL-Divergence Annealing (Prevents posterior collapse).
    - Gradient Clipping (Prevents exploding gradients).
    - Latency Monitoring.
    """
    
    def __init__(self, 
                 latent_dim=128, 
                 beta_max=1.0, 
                 epochs=50, 
                 batch_size=32, 
                 lr=1e-3, 
                 device='cuda'):
        
        self.config = {
            'latent_dim': latent_dim,
            'beta_max': beta_max,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'device': device
        }
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f">> [System] Initializing Trainer on {self.device}")
        
        # 1. Initialize Components
        self.model = VAE(latent_dim=latent_dim).to(self.device)
        
        # 2. Unsupervised Data Pipeline
        self.train_dataset = ShapeNetCore(train=True)
        self.test_dataset = ShapeNetCore(train=False)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # 3. Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = AetherLoss()
        
        # 4. Storage
        os.makedirs("checkpoints", exist_ok=True)
        self.best_loss = float('inf')

    def get_beta(self, epoch):
        """
        Cyclical Annealing Schedule.
        Linearly increases beta from 0 to beta_max to organize latent space.
        """
        # Simple Linear Warmup over first 10 epochs
        warmup = 10
        if epoch < warmup:
            return (epoch / warmup) * self.config['beta_max']
        return self.config['beta_max']

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        beta = self.get_beta(epoch)
        self.loss_fn.beta = beta  # Update loss function dynamic parameter
        
        # Progress Bar
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch in loop:
            # Move data to GPU
            x = batch.to(self.device)
            
            # Forward Pass
            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.model(x)
            
            # Loss Calculation
            loss, recon, kld = self.loss_fn(recon_x, x, mu, logvar)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping (Senior Engineer Stability Trick)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            
            loop.set_postfix(loss=loss.item(), beta=beta)
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                x = batch.to(self.device)
                recon_x, mu, logvar = self.model(x)
                loss, _, _ = self.loss_fn(recon_x, x, mu, logvar)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

    def run(self):
        print(f">> [System] Starting Training for {self.config['epochs']} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            # Checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), "checkpoints/aether_best.pth")
                save_msg = "Model Saved!"
            else:
                save_msg = ""
                
            print(f"   [Summary] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {save_msg}")
            
        total_time = (time.time() - start_time) / 60
        print(f">> [System] Training Complete in {total_time:.2f} mins. Best Loss: {self.best_loss:.4f}")

if __name__ == "__main__":
    trainer = AetherTrainer(epochs=30) # Quick run to verify system
    trainer.run()