"""
Supervised Contrastive Loss Functions
Loss functions for training the contrastive autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss implementation
    
    Pulls samples from the same class together while pushing samples 
    from different classes apart in the latent space.
    
    Args:
        temperature: Temperature parameter for scaling similarities
        base_temperature: Base temperature for normalization
    """
    
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):
        """
        Compute supervised contrastive loss
        
        Args:
            features: Latent features [batch_size, feature_dim]
            labels: Class labels [batch_size]
            
        Returns:
            loss: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features to unit sphere
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label masks
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        mask_negative = 1 - mask_positive
        
        # Remove diagonal (self-similarity)
        mask_positive = mask_positive - torch.eye(batch_size).to(device)
        
        # For numerical stability, subtract max
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean log probability over positive pairs
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / mask_positive.sum(1)
        
        # Loss is negative log probability
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Average over batch, excluding samples with no positive pairs
        valid_samples = mask_positive.sum(1) > 0
        loss = loss[valid_samples].mean()
        
        return loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder
    
    Args:
        loss_type: Type of reconstruction loss ('mse' or 'l1')
    """
    
    def __init__(self, loss_type='mse'):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, reconstructed, original):
        """
        Compute reconstruction loss
        
        Args:
            reconstructed: Reconstructed embeddings [batch_size, input_dim]
            original: Original embeddings [batch_size, input_dim]
            
        Returns:
            loss: Scalar loss value
        """
        return self.loss_fn(reconstructed, original)


class CombinedLoss(nn.Module):
    """
    Combined loss function for contrastive autoencoder
    
    Combines supervised contrastive loss with reconstruction loss
    
    Args:
        contrastive_weight: Weight for contrastive loss component
        reconstruction_weight: Weight for reconstruction loss component
        temperature: Temperature for contrastive loss
        reconstruction_type: Type of reconstruction loss
    """
    
    def __init__(self, contrastive_weight=1.0, reconstruction_weight=1.0, 
                 temperature=0.1, reconstruction_type='mse'):
        super(CombinedLoss, self).__init__()
        
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.reconstruction_loss = ReconstructionLoss(loss_type=reconstruction_type)

    @staticmethod
    def get_contrastive_beta(epoch, warmup_epochs=10, max_beta=2.0, schedule_type='linear'):
        """
        Beta scheduling for contrastive loss weight
        
        Args:
            epoch: Current epoch number
            warmup_epochs: Number of epochs with pure reconstruction (β=0)
            max_beta: Maximum contrastive weight
            schedule_type: 'linear', 'cosine', or 'exponential'
        """
        if epoch < warmup_epochs:
            return 0.0  # Pure reconstruction phase
        
        # Calculate progress after warmup
        progress = (epoch - warmup_epochs) / max(1, (50 - warmup_epochs))  # Assuming 50 total epochs
        progress = min(1.0, progress)  # Clamp to [0, 1]
        
        if schedule_type == 'linear':
            return progress * max_beta
        elif schedule_type == 'cosine':
            import math
            return 0.5 * max_beta * (1 + math.cos(math.pi * (1 - progress)))
        elif schedule_type == 'exponential':
            return max_beta * (progress ** 2)
        else:
            return progress * max_beta

    def forward(self, latent_features, labels, reconstructed, original, contrastive_weight=None):
        """
        Compute combined loss
        Args:
            latent_features: Latent representations [batch_size, latent_dim]
            labels: Class labels [batch_size]
            reconstructed: Reconstructed embeddings [batch_size, input_dim]
            original: Original embeddings [batch_size, input_dim]
            contrastive_weight: Beta parameter for scheduling
        Returns:
            total_loss: Combined loss value
            contrastive_loss: Contrastive loss component
            reconstruction_loss: Reconstruction loss component
        """
        if contrastive_weight is None:
            contrastive_weight = self.contrastive_weight
        # Use dynamic weight instead of fixed weight
        contrastive_loss = self.contrastive_loss(latent_features, labels)
        reconstruction_loss = self.reconstruction_loss(reconstructed, original)
        
        total_loss = (contrastive_weight * contrastive_loss + 
                     self.reconstruction_weight * reconstruction_loss)
        
        return total_loss, contrastive_loss, reconstruction_loss
    

def test_losses():
    """Test loss function implementations"""
    print("Testing Loss Functions")
    print("=" * 40)
    
    # Create test data
    batch_size = 32
    latent_dim = 75
    input_dim = 768
    
    # Generate random features and labels
    latent_features = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, 3, (batch_size,))  # 3 classes: 0, 1, 2
    original_embeddings = torch.randn(batch_size, input_dim)
    reconstructed_embeddings = torch.randn(batch_size, input_dim)
    
    print(f"Latent features shape: {latent_features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Original embeddings shape: {original_embeddings.shape}")
    print(f"Reconstructed embeddings shape: {reconstructed_embeddings.shape}")
    print()
    
    # Test individual losses
    print("Testing Individual Loss Components:")
    print("-" * 40)
    
    # Contrastive loss
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.1)
    contrastive_loss = contrastive_loss_fn(latent_features, labels)
    print(f"Contrastive Loss: {contrastive_loss.item():.4f}")
    
    # Reconstruction loss
    reconstruction_loss_fn = ReconstructionLoss(loss_type='mse')
    reconstruction_loss = reconstruction_loss_fn(reconstructed_embeddings, original_embeddings)
    print(f"Reconstruction Loss: {reconstruction_loss.item():.4f}")
    
    # Combined loss
    print("\nTesting Combined Loss:")
    print("-" * 40)
    
    combined_loss_fn = CombinedLoss(
        contrastive_weight=1.0,
        reconstruction_weight=1.0,
        temperature=0.1
    )
    
    total_loss, cont_loss, recon_loss = combined_loss_fn(
        latent_features, labels, reconstructed_embeddings, original_embeddings
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Contrastive Component: {cont_loss.item():.4f}")
    print(f"Reconstruction Component: {recon_loss.item():.4f}")
    
    # Test gradient flow
    print("\nTesting Gradient Flow:")
    print("-" * 40)
    
    # Create simple model for gradient test
    test_model = nn.Linear(latent_dim, latent_dim)
    test_features = test_model(latent_features)
    
    # Compute loss and backpropagate
    loss, _, _ = combined_loss_fn(test_features, labels, reconstructed_embeddings, original_embeddings)
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in test_model.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    if has_gradients:
        grad_norms = [p.grad.norm().item() for p in test_model.parameters() if p.grad is not None]
        print(f"Gradient norms: {grad_norms}")
    
    print("\nTesting different class distributions:")
    print("-" * 40)
    
    # Test with different label distributions
    test_cases = [
        torch.zeros(batch_size, dtype=torch.long),  # All same class
        torch.randint(0, 3, (batch_size,)),         # Random distribution
        torch.cat([torch.zeros(10), torch.ones(10), torch.full((12,), 2)]).long()  # Balanced
    ]
    
    for i, test_labels in enumerate(test_cases):
        try:
            loss = contrastive_loss_fn(latent_features, test_labels)
            print(f"Test case {i+1}: Loss = {loss.item():.4f}")
        except Exception as e:
            print(f"Test case {i+1}: Error - {e}")
    
    print("\nLoss function tests completed!")


if __name__ == "__main__":
    test_losses()
