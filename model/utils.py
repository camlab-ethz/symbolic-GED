import os
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
import h5py
from typing import Dict, Optional
from model.model import GrammarVAE


class AnnealKL:
    """KL Annealing with step and rate control."""
    def __init__(self, step=1e-3, rate=500):  # Tighter KL annealing
        self.step = step
        self.rate = rate

    def alpha(self, update: int) -> float:
        return min(1, (update // self.rate) * self.step)


def load_data(data_path: str) -> TensorDataset:
    """Load data from HDF5 file as inputs and targets."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with h5py.File(data_path, 'r') as f:
        data = torch.from_numpy(f['data'][:]).float()  # Shape: [batch, seq_length, features]
    data = data.transpose(1, 2)  # Align to [batch, features, seq_length]
    targets = data.argmax(1)  # Convert to class indices: [batch, seq_length]
    return TensorDataset(data, targets)


class GrammarVAEModel(pl.LightningModule):
    """PyTorch Lightning module for Grammar VAE."""
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()

        self.model = GrammarVAE(config)
        self.criterion = nn.CrossEntropyLoss()
        self.anneal = AnnealKL(step=1e-6, rate=500)
        self.lr = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.data_path = config['data']['data_path']
        self.validation_split = config['data']['validation_split']

    def setup(self, stage: Optional[str] = None):
        dataset = load_data(self.data_path)  # Targets are class indices
        val_size = int(self.validation_split * len(dataset))
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    
    def forward(self, x):
        """Forward pass through the VAE."""
        mu, sigma = self.model.encoder(x)
        z = self.model.sample(mu, sigma, num_samples=10)  # specify number of samples
        logits = self.model.decoder(z)
        return logits, mu, sigma
    def compute_loss(self, logits, targets, mu, sigma, step):
        """
        Compute reconstruction loss, KL loss, and ELBO.
        Args:
            logits: shape (batch_size, num_samples, max_length, num_classes)
            targets: shape (batch_size, max_length)
        """
        batch_size, num_samples, seq_length, num_classes = logits.shape
        
        # Average over samples dimension first
        logits = torch.mean(logits, dim=1)  # (batch_size, max_length, num_classes)
        
        # Reshape for cross entropy loss
        logits = logits.view(-1, num_classes)  # (batch_size * max_length, num_classes)
        targets = targets.view(-1)  # (batch_size * max_length)
        
        # Compute reconstruction loss
        rec_loss = self.criterion(logits, targets)
        
        # Compute KL loss (doesn't change as it's analytical)
        kl_loss = self.model.kl_divergence(mu, sigma)
        alpha = self.anneal.alpha(step)
        
        return rec_loss + alpha * kl_loss, rec_loss, kl_loss
   
    def compute_sequence_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute per-sequence accuracy: A sequence is correct if all tokens match.
        Args:
            logits: Tensor of shape [batch_size, num_samples, seq_length, num_classes] 
                or [batch_size, seq_length, num_classes]
            targets: Tensor of shape [batch_size, seq_length]
        Returns:
            float: Per-sequence accuracy as a percentage
        """
        # If we have multiple samples, take the mean over samples first
        if len(logits.shape) == 4:
            logits = torch.mean(logits, dim=1)  # Average over samples dimension
            
        # Get predictions
        preds = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_length]
        
        # Check if all tokens match across each sequence
        correct = (preds == targets).all(dim=1)  # Shape: [batch_size]
        
        # Compute the mean accuracy over the batch
        sequence_accuracy = correct.float().mean().item() * 100  # Convert to percentage
        
        return sequence_accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, mu, sigma = self(x)
        if batch_idx == 0:
            print(f"Batch {batch_idx} - Logits shape: {logits.shape}")
            print(f"Batch {batch_idx} - Targets shape: {y.shape}")
            
            preds = torch.argmax(logits, dim=-1)
            print("Predicted Indices (Logits argmax):")
            print(preds)
            
            print("True Targets:")
            print(y)
            
            accuracy = self.compute_sequence_accuracy(logits, y)
            print(f"Per-sequence Accuracy (Batch {batch_idx}): {accuracy:.2f}%")

        # Compute losses
        elbo, rec_loss, kl_loss = self.compute_loss(logits, y, mu, sigma, self.global_step)
        
        # Compute per-sequence accuracy
        
        acc = self.compute_sequence_accuracy(logits, y)

        self.log_dict({
            'train_loss': rec_loss,
            'train_kl': kl_loss,
            'train_elbo': elbo,
            'train_acc': acc
        }, prog_bar=True)
        return elbo

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, mu, sigma = self(x)
       # Debugging: Print shapes, content, and accuracy for the first iteration
        if batch_idx == 0:
            print(f"Batch {batch_idx} - Logits shape: {logits.shape}")
            print(f"Batch {batch_idx} - Targets shape: {y.shape}")
            
            preds = torch.argmax(logits, dim=-1)
            print("Predicted Indices (Logits argmax):")
            print(preds)
            
            print("True Targets:")
            print(y)
            
            accuracy = self.compute_sequence_accuracy(logits, y)
            print(f"Per-sequence Accuracy (Batch {batch_idx}): {accuracy:.2f}%")
            

        # Compute losses
        elbo, rec_loss, kl_loss = self.compute_loss(logits, y, mu, sigma, self.global_step)
        
        # Compute per-sequence accuracy
        
        acc = self.compute_sequence_accuracy(logits, y)

        self.log_dict({
            'val_loss': rec_loss,
            'val_kl': kl_loss,
            'val_elbo': elbo,
            'val_acc': acc
        }, prog_bar=True)
        return elbo

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_elbo"}}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3)
