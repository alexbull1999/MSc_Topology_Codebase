"""
Test contrastive loss with your actual model and data
Add this to your project and run it to diagnose the real issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contrastive_autoencoder_model import ContrastiveAutoencoder
from losses import SupervisedContrastiveLoss

def test_real_model_contrastive():
    """Test contrastive loss with actual model architecture"""
    print("Testing contrastive loss with real model...")
    print("=" * 50)
    
    # Create your actual model
    model = ContrastiveAutoencoder(
        input_dim=768,
        latent_dim=75,
        hidden_dims=[512, 256],
        dropout_rate=0.2
    )
    model.eval()  # Set to eval to remove dropout randomness
    
    # Create synthetic lattice containment embeddings that mimic your real data
    # But with KNOWN class structure
    batch_size = 30
    
    # Create 3 distinct "lattice containment" patterns
    # Class 0 (entailment): high positive values in first 256 dims
    class_0_base = torch.cat([
        torch.ones(256) * 0.8,  # High containment
        torch.zeros(256),       # Medium containment  
        torch.ones(256) * -0.2  # Low containment
    ])
    
    # Class 1 (neutral): medium values everywhere
    class_1_base = torch.ones(768) * 0.3
    
    # Class 2 (contradiction): inverse pattern of class 0
    class_2_base = torch.cat([
        torch.ones(256) * -0.8, # Low containment
        torch.zeros(256),       # Medium containment
        torch.ones(256) * 0.8   # High containment  
    ])
    
    # Create batches with noise
    embeddings_list = []
    labels_list = []
    
    for class_idx, base_pattern in enumerate([class_0_base, class_1_base, class_2_base]):
        # 10 samples per class (like your balanced batches)
        for i in range(10):
            # Add noise but keep class structure
            noisy_embedding = base_pattern + torch.randn(768) * 0.1
            embeddings_list.append(noisy_embedding)
            labels_list.append(class_idx)
    
    # Stack into tensors
    embeddings = torch.stack(embeddings_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    print(f"Created synthetic data:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels: {labels}")
    print(f"  Class distribution: {torch.bincount(labels)}")
    
    # Test what your encoder does to this data
    print(f"\nTesting encoder behavior...")
    with torch.no_grad():
        latent_features, reconstructed = model(embeddings)
    
    print(f"Latent features shape: {latent_features.shape}")
    print(f"Latent features stats:")
    print(f"  Min: {latent_features.min().item():.4f}")
    print(f"  Max: {latent_features.max().item():.4f}")
    print(f"  Mean: {latent_features.mean().item():.4f}")
    print(f"  Std: {latent_features.std().item():.4f}")
    
    # Check if encoder is producing meaningful class separation
    class_means = []
    for class_idx in range(3):
        class_mask = labels == class_idx
        class_latent = latent_features[class_mask]
        class_mean = class_latent.mean(dim=0)
        class_means.append(class_mean)
        
        print(f"\nClass {class_idx} latent stats:")
        print(f"  Mean vector norm: {class_mean.norm().item():.4f}")
        print(f"  Within-class std: {class_latent.std().item():.4f}")
    
    # Check inter-class distances
    print(f"\nInter-class distances (in latent space):")
    for i in range(3):
        for j in range(i+1, 3):
            distance = torch.norm(class_means[i] - class_means[j]).item()
            print(f"  Class {i} <-> Class {j}: {distance:.4f}")
    
    # Now test contrastive loss with different temperatures
    print(f"\nTesting contrastive loss with different temperatures:")
    temperatures = [1.0, 0.5, 0.1, 0.05]
    
    for temp in temperatures:
        loss_fn = SupervisedContrastiveLoss(temperature=temp)
        loss = loss_fn(latent_features, labels)
        
        print(f"  Temperature {temp}: Loss = {loss.item():.4f}")
        
        # Analyze similarities at this temperature
        features_norm = F.normalize(latent_features, dim=1)
        sim_matrix = torch.matmul(features_norm, features_norm.T) / temp
        
        # Check positive vs negative similarities
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float()
        mask_no_diag = 1 - torch.eye(len(labels))
        mask_positive = mask_positive * mask_no_diag
        
        pos_sims = sim_matrix[mask_positive.bool()]
        neg_sims = sim_matrix[(1 - mask_positive).bool()]
        
        if len(pos_sims) > 0 and len(neg_sims) > 0:
            separation = (pos_sims.mean() - neg_sims.mean()).item()
            print(f"    Pos sims: {pos_sims.mean().item():.4f} ± {pos_sims.std().item():.4f}")
            print(f"    Neg sims: {neg_sims.mean().item():.4f} ± {neg_sims.std().item():.4f}")
            print(f"    Separation: {separation:.4f}")
        else:
            print(f"    ERROR: No positive or negative pairs found!")
    
    # Test if the model can learn on this synthetic data
    print(f"\nTesting if model can learn on synthetic data...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SupervisedContrastiveLoss(temperature=0.5)  # Use reasonable temperature
    
    initial_loss = None
    for epoch in range(20):  # Quick training test
        optimizer.zero_grad()
        latent_features, reconstructed = model(embeddings)
        loss = loss_fn(latent_features, labels)
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    improvement = initial_loss - final_loss
    print(f"\nLearning test results:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    
    if improvement > 0.1:
        print("  ✓ Model CAN learn on synthetic data - issue might be with real data")
    else:
        print("  ✗ Model CANNOT learn even on synthetic data - issue is with model/loss")

if __name__ == "__main__":
    test_real_model_contrastive()