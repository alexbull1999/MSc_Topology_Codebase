"""
Loss Functions for Global Dataset Contrastive Training
Clean implementation with full dataset contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FullDatasetContrastiveLoss(nn.Module):
    """
    Contrastive loss that uses the entire dataset for global context
    """
    
    def __init__(self, margin=2.0, update_frequency=3, max_global_samples=5000):
        super().__init__()
        self.margin = margin
        self.update_frequency = update_frequency
        self.max_global_samples = max_global_samples
        self.global_features = None
        self.global_labels = None
        self.batch_count = 0
        
        print(f"FullDatasetContrastiveLoss initialized:")
        print(f"  Margin: {margin}")
        print(f"  Update frequency: {update_frequency} epochs")
        print(f"  Max global samples: {max_global_samples}")
    
    def update_global_dataset(self, dataloader, model, device):
        """
        Extract features for the entire dataset
        """
        print("🌍 Extracting features for entire dataset...")
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                embeddings = batch['embeddings'].to(device)
                labels = batch['labels'].to(device)
                
                # Get latent features
                latent, _ = model(embeddings)
                
                all_features.append(latent.cpu())
                all_labels.append(labels.cpu())
                
                if batch_idx % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Combine all features
        full_features = torch.cat(all_features, dim=0).to(device)
        full_labels = torch.cat(all_labels, dim=0).to(device)
        
        # Subsample if too large
        if len(full_features) > self.max_global_samples:
            indices = torch.randperm(len(full_features))[:self.max_global_samples]
            self.global_features = full_features[indices]
            self.global_labels = full_labels[indices]
            print(f"  Subsampled to {self.max_global_samples} samples")
        else:
            self.global_features = full_features
            self.global_labels = full_labels

        
        print(f"  Global dataset updated: {self.global_features.shape[0]} samples")
        
        # Analyze global separation
        self._analyze_global_separation()
        
        model.train()
    
    def _analyze_global_separation(self):
        """
        Analyze separation in the global dataset
        """
        if self.global_features is None:
            return
        
        print("  Analyzing global separation...")
        distances = torch.cdist(self.global_features, self.global_features, p=2)
        
        labels_expanded = self.global_labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float()
        mask_no_diagonal = 1 - torch.eye(len(self.global_labels), device=self.global_features.device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float()) * mask_no_diagonal
        
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) > 0 and len(neg_distances) > 0:
            pos_mean = pos_distances.mean().item()
            neg_mean = neg_distances.mean().item()
            ratio = neg_mean / pos_mean
            gap = neg_distances.min().item() - pos_distances.max().item()
            
            print(f"  GLOBAL ANALYSIS:")
            print(f"    Pos distances: {pos_mean:.3f} ± {pos_distances.std().item():.3f}")
            print(f"    Neg distances: {neg_mean:.3f} ± {neg_distances.std().item():.3f}")
            print(f"    Separation ratio: {ratio:.2f}x")
            print(f"    Gap: {gap:.3f}")
            
            if ratio > 3.0:
                print("    ✅ Excellent global separation!")
            elif ratio > 2.0:
                print("    ✅ Good global separation")
            elif ratio > 1.5:
                print("    ⚠️  Moderate global separation")
            else:
                print("    ❌ Poor global separation")
    
    def forward(self, features, labels):
        """
        Compute contrastive loss using global dataset context
        """
        self.batch_count += 1
        current_batch_size = features.shape[0]
        
        if self.global_features is None:
            # Fallback to current batch if global not available
            print(f"  Using current batch only: {current_batch_size} samples")
            return self._compute_contrastive_loss(features, labels)
        
        # Sample from global dataset for efficiency
        global_size = self.global_features.shape[0]
        if global_size > 1500:  # Sample for computational efficiency
            indices = torch.randperm(global_size)[:1500]
            global_features_subset = self.global_features[indices]
            global_labels_subset = self.global_labels[indices]
        else:
            global_features_subset = self.global_features
            global_labels_subset = self.global_labels
        
        # Combine current batch with global samples
        all_features = torch.cat([features, global_features_subset], dim=0)
        all_labels = torch.cat([labels, global_labels_subset], dim=0)
        
        print(f"  Enhanced batch: {current_batch_size} current + {len(global_features_subset)} global = {len(all_features)} total")
        
        return self._compute_contrastive_loss(all_features, all_labels)
    
    def _compute_contrastive_loss(self, features, labels):
        """
        Compute proper contrastive loss on feature set
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)
        
        # Create masks
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float().to(device)) * mask_no_diagonal
        
        # Get positive and negative distances
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Proper contrastive loss: minimize positive, maximize negative (with margin)
        pos_loss = pos_distances.mean()
        neg_loss = torch.clamp(self.margin - neg_distances, min=0).mean()
        
        total_loss = pos_loss + neg_loss
        
        # Debug current batch separation
        if len(pos_distances) > 10:  # Only print if reasonable sample size
            separation_ratio = neg_distances.mean() / pos_distances.mean()
            gap = neg_distances.min() - pos_distances.max()
            print(f"    Batch distances: Pos={pos_distances.mean():.3f}, Neg={neg_distances.mean():.3f}, Ratio={separation_ratio:.2f}x, Gap={gap:.3f}")
        
        return total_loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder
    """
    
    def __init__(self, loss_type='mse'):
        super().__init__()
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
        """
        return self.loss_fn(reconstructed, original)


class FullDatasetCombinedLoss(nn.Module):
    """
    Combined loss using full dataset contrastive loss
    """
    
    def __init__(self, contrastive_weight=1.0, reconstruction_weight=0.0, 
                 margin=2.0, update_frequency=3, max_global_samples=5000):
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.contrastive_loss = FullDatasetContrastiveLoss(
            margin=margin, 
            update_frequency=update_frequency,
            max_global_samples=max_global_samples
        )
        
        if reconstruction_weight > 0:
            self.reconstruction_loss = ReconstructionLoss()
        else:
            self.reconstruction_loss = None
        
        print(f"FullDatasetCombinedLoss initialized:")
        print(f"  Contrastive weight: {contrastive_weight}")
        print(f"  Reconstruction weight: {reconstruction_weight}")
    
    def forward(self, latent_features, labels, reconstructed=None, original=None, contrastive_weight=None):
        """
        Compute combined loss with full dataset contrastive
        """
        if contrastive_weight is None:
            contrastive_weight = self.contrastive_weight
        
        # Full dataset contrastive loss
        contrastive_loss = self.contrastive_loss(latent_features, labels)
        
        # Reconstruction loss
        if self.reconstruction_loss is not None and reconstructed is not None and original is not None:
            reconstruction_loss = self.reconstruction_loss(reconstructed, original)
        else:
            reconstruction_loss = torch.tensor(0.0, device=latent_features.device)
        
        # Combined loss
        total_loss = (contrastive_weight * contrastive_loss + 
                     self.reconstruction_weight * reconstruction_loss)
        
        return total_loss, contrastive_loss, reconstruction_loss
    
    def get_distance_stats(self, latent_features, labels):
        """
        Get distance statistics for debugging (compatibility method)
        """
        batch_size = latent_features.shape[0]
        device = latent_features.device
        
        distances = torch.cdist(latent_features, latent_features, p=2)
        
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float().to(device)) * mask_no_diagonal
        
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return {
                'pos_mean': 0.0, 'pos_std': 0.0, 'pos_min': 0.0, 'pos_max': 0.0,
                'neg_mean': 0.0, 'neg_std': 0.0, 'neg_min': 0.0, 'neg_max': 0.0,
                'separation_ratio': 0.0, 'gap': 0.0
            }
        
        stats = {
            'pos_mean': pos_distances.mean().item(),
            'pos_std': pos_distances.std().item(),
            'pos_min': pos_distances.min().item(),
            'pos_max': pos_distances.max().item(),
            'neg_mean': neg_distances.mean().item(),
            'neg_std': neg_distances.std().item(),
            'neg_min': neg_distances.min().item(),
            'neg_max': neg_distances.max().item(),
            'separation_ratio': (neg_distances.mean() / pos_distances.mean()).item(),
            'gap': (neg_distances.min() - pos_distances.max()).item()
        }
        
        return stats


def test_losses():
    """Test loss functions with synthetic data"""
    print("Testing loss functions...")
    
    # Create synthetic data
    batch_size = 64
    latent_dim = 75
    
    # Create features with some class structure
    features = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, 3, (batch_size,))
    
    print(f"Test data: {features.shape}, labels: {torch.unique(labels, return_counts=True)}")
    
    # Test FullDatasetCombinedLoss
    loss_fn = FullDatasetCombinedLoss(
        contrastive_weight=1.0,
        reconstruction_weight=0.0,
        margin=2.0,
        update_frequency=3
    )
    
    # Test without global dataset (should fallback to current batch)
    reconstructed = torch.randn_like(features)
    original = torch.randn_like(features)
    
    total_loss, contrastive_loss, reconstruction_loss = loss_fn(
        features, labels, reconstructed, original
    )
    
    print(f"Loss test results:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Contrastive loss: {contrastive_loss.item():.4f}")
    print(f"  Reconstruction loss: {reconstruction_loss.item():.4f}")
    
    # Test distance stats
    stats = loss_fn.get_distance_stats(features, labels)
    print(f"Distance stats: Ratio={stats['separation_ratio']:.2f}x, Gap={stats['gap']:.3f}")
    
    print("✅ Loss functions test completed!")


if __name__ == "__main__":
    test_losses()