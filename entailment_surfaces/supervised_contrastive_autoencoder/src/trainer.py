"""
Training Pipeline Module for Supervised Contrastive Autoencoder
Handles model training, validation, and checkpoint management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from contrastive_autoencoder_model import ContrastiveAutoencoder
from losses import CombinedLoss
from data_loader import EntailmentDataLoader


class ContrastiveAutoencoderTrainer:
    """
    Trainer for the supervised contrastive autoencoder
    """
    
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        """
        Initialize trainer
        
        Args:
            model: ContrastiveAutoencoder instance
            loss_function: CombinedLoss instance
            optimizer: PyTorch optimizer
            device: Device to train on
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_contrastive_loss': [],
            'train_reconstruction_loss': [],
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_reconstruction_loss': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"Trainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader, current_epoch, beta_config=None):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            current_epoch: Current epoch number (for beta scheduling)
            beta_config: Beta scheduling configuration
            
        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.train()

        # Calculate dynamic contrastive weight if beta scheduling is enabled
        if beta_config and beta_config.get('enabled', False):
            from losses import CombinedLoss
            contrastive_weight = CombinedLoss.get_contrastive_beta(
                epoch=current_epoch,
                warmup_epochs=beta_config.get('warmup_epochs', 10),
                max_beta=beta_config.get('max_beta', 2.0),
                schedule_type=beta_config.get('schedule_type', 'linear'),
                total_epochs=beta_config.get('total_epochs', 50) 
            )
        else:
            contrastive_weight = None  # Use default from loss function
        
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'contrastive_weight': contrastive_weight if contrastive_weight is not None else self.loss_function.contrastive_weight,
            'num_batches': 0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            latent, reconstructed = self.model(embeddings)
            
            # Compute loss
            total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                latent, labels, reconstructed, embeddings, contrastive_weight=contrastive_weight
            )

            # Check for NaN loss immediately after calculation
            if torch.isnan(total_loss):
                print(f"!!! NaN loss detected at batch {batch_idx}. Stopping epoch. !!!")
                print(f"  Contrastive Loss: {contrastive_loss.item()}, Recon Loss: {reconstruction_loss.item()}")
                # Break the loop to prevent NaN from propagating
                # We will return the averages of the batches that successfully completed
                exit(1)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['contrastive_loss'] += contrastive_loss.item()
            epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
            epoch_losses['num_batches'] += 1
            
            # Print progress
            if batch_idx % 100 == 0:
                beta_info = f", β={contrastive_weight:.3f}" if contrastive_weight is not None else ""
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss = {total_loss.item():.4f} "
                      f"(C: {contrastive_loss.item():.4f}, "
                      f"R: {reconstruction_loss.item():.4f}{beta_info})")
        
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches'],
            'contrastive_weight': epoch_losses['contrastive_weight']
        }
        
        return avg_losses
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with average validation losses
        """
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'num_batches': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                latent, reconstructed = self.model(embeddings)
                
                # Compute loss
                total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                    latent, labels, reconstructed, embeddings
                )
                
                # Accumulate losses
                epoch_losses['total_loss'] += total_loss.item()
                epoch_losses['contrastive_loss'] += contrastive_loss.item()
                epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
                epoch_losses['num_batches'] += 1
        
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches']
        }
        
        return avg_losses
    
    def train(self, train_loader, val_loader, num_epochs=50, patience=10, 
              save_dir='checkpoints', save_every=5, beta_config=None):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            beta_config: Beta scheduling configuration
        """
        print("Starting training...")
        if beta_config and beta_config.get('enabled', False):
            print("Beta scheduling enabled:")
            print(f"  Warmup epochs: {beta_config.get('warmup_epochs', 10)}")
            print(f"  Max beta: {beta_config.get('max_beta', 2.0)}")
            print(f"  Schedule type: {beta_config.get('schedule_type', 'linear')}")
        print("=" * 50)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Add beta tracking to training history
        if 'contrastive_weight' not in self.train_history:
            self.train_history['contrastive_weight'] = []
        
        # Store beta_config for use throughout training
        self.beta_config = beta_config
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)
            
            # Training phase
            train_losses = self.train_epoch(train_loader, epoch, self.beta_config)
            
            # Validation phase (always use current beta for consistency)
            val_losses = self.validate_epoch(val_loader)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            beta_info = f" (β={train_losses['contrastive_weight']:.3f})" if self.beta_config and self.beta_config.get('enabled', False) else ""
            print(f"Train Loss: {train_losses['total_loss']:.4f} "
                  f"(C: {train_losses['contrastive_loss']:.4f}, "
                  f"R: {train_losses['reconstruction_loss']:.4f}){beta_info}")
            print(f"Val Loss: {val_losses['total_loss']:.4f} "
                  f"(C: {val_losses['contrastive_loss']:.4f}, "
                  f"R: {val_losses['reconstruction_loss']:.4f})")
            
            # Update training history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_losses['total_loss'])
            self.train_history['train_contrastive_loss'].append(train_losses['contrastive_loss'])
            self.train_history['train_reconstruction_loss'].append(train_losses['reconstruction_loss'])
            self.train_history['val_loss'].append(val_losses['total_loss'])
            self.train_history['val_contrastive_loss'].append(val_losses['contrastive_loss'])
            self.train_history['val_reconstruction_loss'].append(val_losses['reconstruction_loss'])
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.train_history['contrastive_weight'].append(train_losses['contrastive_weight'])
            
            # Check for improvement
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(save_dir, f'best_model.pt', epoch + 1, is_best=True)
                print(f"New best model saved (Val Loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir, f'checkpoint_epoch_{epoch + 1}.pt', epoch + 1)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best model was at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.4f}")
                break
        
        print("\nTraining completed!")
        print(f"Best model: Epoch {self.best_epoch}, Val Loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(save_dir, 'final_model.pt', epoch + 1, is_final=True)
        
        # Save training history
        self.save_training_history(save_dir)
    
    def save_checkpoint(self, save_dir, filename, epoch, is_best=False, is_final=False):
        """
        Save model checkpoint
        
        Args:
            save_dir: Directory to save checkpoint
            filename: Checkpoint filename
            epoch: Current epoch
            is_best: Whether this is the best model
            is_final: Whether this is the final model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'hidden_dims': self.model.hidden_dims,
                'dropout_rate': self.model.dropout_rate
            }
        }
        
        if is_best:
            checkpoint['is_best'] = True
        if is_final:
            checkpoint['is_final'] = True
        
        filepath = os.path.join(save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def save_training_history(self, save_dir):
        """
        Save training history as JSON
        
        Args:
            save_dir: Directory to save history
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(save_dir, f'training_history_{timestamp}.json')
        
        with open(history_file, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"Training history saved: {history_file}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint['train_history']
        
        print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}, "
              f"Best Val Loss: {self.best_val_loss:.4f}")
        
        return checkpoint['epoch']


def create_trainer(model_config, loss_config, optimizer_config, device='cuda'):
    """
    Factory function to create trainer with all components
    
    Args:
        model_config: Dictionary with model configuration
        loss_config: Dictionary with loss configuration
        optimizer_config: Dictionary with optimizer configuration
        device: Device to use for training
        
    Returns:
        Configured trainer instance
    """
    # Create model
    model = ContrastiveAutoencoder(**model_config)
    
    # Create loss function
    loss_fn = CombinedLoss(**loss_config)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), **optimizer_config)
    
    # Create trainer
    trainer = ContrastiveAutoencoderTrainer(model, loss_fn, optimizer, device)
    
    return trainer
    


def test_trainer():
    """Test trainer functionality with mock components"""
    print("Testing Trainer Module")
    print("=" * 40)
    
    # Create model
    print("Creating model...")
    model = ContrastiveAutoencoder(
        input_dim=768,
        latent_dim=75,
        hidden_dims=[512, 256],
        dropout_rate=0.2
    )
    
    # Create loss function
    print("Creating loss function...")
    loss_fn = CombinedLoss(
        contrastive_weight=1.0,
        reconstruction_weight=1.0,
        temperature=0.1
    )
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    print("Creating trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ContrastiveAutoencoderTrainer(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        device=device
    )

    print(f"Trainer created successfully on {device}")
    
    # Test with synthetic data
    print("\nTesting with synthetic data...")
    batch_size = 32
    num_batches = 5
    
    # Create synthetic dataset
    synthetic_embeddings = torch.randn(batch_size * num_batches, 768)
    synthetic_labels = torch.randint(0, 3, (batch_size * num_batches,))
    
    # Create simple dataset and dataloader
    from torch.utils.data import TensorDataset, DataLoader
    
    synthetic_dataset = TensorDataset(synthetic_embeddings, synthetic_labels)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)
    
    # Convert to expected format
    class SyntheticDataset:
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels
        
        def __len__(self):
            return len(self.embeddings)
        
        def __getitem__(self, idx):
            return {
                'embeddings': self.embeddings[idx],
                'labels': self.labels[idx]
            }
    
    synthetic_dataset = SyntheticDataset(synthetic_embeddings, synthetic_labels)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created synthetic dataset with {len(synthetic_dataset)} samples")
    
    # Test single training epoch
    print("\nTesting single training epoch...")
    train_losses = trainer.train_epoch(synthetic_loader)
    
    print(f"Training losses: {train_losses}")
    
    # Test validation epoch
    print("\nTesting validation epoch...")
    val_losses = trainer.validate_epoch(synthetic_loader)
    
    print(f"Validation losses: {val_losses}")
    
    # Test checkpoint saving
    print("\nTesting checkpoint saving...")
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer.save_checkpoint(temp_dir, 'test_checkpoint.pt', epoch=1)
        
        # Test checkpoint loading
        print("Testing checkpoint loading...")
        loaded_epoch = trainer.load_checkpoint(os.path.join(temp_dir, 'test_checkpoint.pt'))
        print(f"Loaded checkpoint from epoch: {loaded_epoch}")
    
    # Test short training run
    print("\nTesting short training run...")
    trainer.train(
        train_loader=synthetic_loader,
        val_loader=synthetic_loader,
        num_epochs=3,
        patience=2,
        save_dir='test_checkpoints',
        save_every=2
    )
    
    print("\nTrainer testing completed successfully!")
    print("\nTraining history summary:")
    print(f"Epochs trained: {len(trainer.train_history['epoch'])}")
    print(f"Final train loss: {trainer.train_history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {trainer.train_history['val_loss'][-1]:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    
    # Clean up test files
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
        print("Cleaned up test checkpoint directory")


if __name__ == "__main__":
    test_trainer()