import torch
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import matplotlib.pyplot as plt
from ripser import ripser
from gph.python import ripser_parallel
from typing import Tuple, List
import os
import sys
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from attention_autoencoder_model import AttentionAutoencoder
from data_loader_global import GlobalDataLoader



def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                           min_points=200,
                                           max_points=1000,
                                           point_jump=50,
                                           h_dim=0,
                                           alpha: float = 1.) -> Tuple[float, List[np.ndarray]]:
    """
    Compute both PH dimension and persistence diagrams from distance matrix
    This is the CORRECT method used throughout the project
    """
    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape
    
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    all_diagrams = []
    
    for points_number in test_n:
        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]
        
        # Compute persistence diagrams (both H0 and H1)
        result = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")
        diagrams = result['dgms']
        
        # Store diagrams for this sample size
        all_diagrams.append({
            'n_points': points_number,
            'H0': diagrams[0],
            'H1': diagrams[1]
        })
        
        # Compute persistence lengths for PH dimension calculation
        d = diagrams[h_dim]
        d = d[d[:, 1] < np.inf]
        lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())
    
    lengths = np.array(lengths)
    
    # Compute PH dimension
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    phd_score = alpha / (1 - m)
    
    # Return the diagrams from the largest sample size (most stable)
    final_diagrams = all_diagrams[-1]  # Last one has max_points
    
    return phd_score, [final_diagrams['H0'], final_diagrams['H1']]


class BestModelDiagnostic:
    def __init__(self, model_path, data_paths, prototype_path=None, embedding_type='concat'):
        """
        Args:
            model_path: Path to your best 83.13% model checkpoint
            data_paths: Dict with 'train', 'val', 'test' paths to data files
            prototype_path: Path to your prototype diagrams (optional)
            embedding_type: Type of embeddings used ('cosine_concat', etc.)
        """
        self.model_path = model_path
        self.data_paths = data_paths
        self.prototype_path = prototype_path
        self.embedding_type = embedding_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """
        Load the saved contrastive autoencoder model
    
        Args:
            model_path: Path to saved model checkpoint
            device: Device to load model on
        
        Returns:
            Loaded model instance
        """
        print(f"Loading model from: {self.model_path}")
    
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
    
        # Extract model configuration (you may need to adjust these based on your saved model)
        model_config = {
            'input_dim': 1536,  # SBERT concat dimension or Lattice = 768
            'latent_dim': 100,
            'hidden_dims': [1024, 768, 512, 256, 128],
            'dropout_rate': 0.2
        }
        # Create model instance
        model = AttentionAutoencoder(**model_config)
    
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
    
        print(f"Model loaded successfully!")
        print(f"   Best epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
        return model


    def load_validation_data(self):
        """Load validation data using the project's data format"""
        print(f"Loading validation data from {self.data_paths['val']}")
        
        # Load the torch data file (as used in the project)
        data = torch.load(self.data_paths['val'], weights_only=False)
        print(f"Loaded keys: {list(data.keys())}")
        print(f"Total samples: {len(data['labels'])}")
        
        # Generate embeddings using the project's method
        from data_loader_global import GlobalDataLoader
        
        # Create a temporary data loader to generate embeddings
        temp_loader = GlobalDataLoader(
            train_path=self.data_paths['train'],
            val_path=self.data_paths['val'], 
            test_path=self.data_paths['test'],
            embedding_type=self.embedding_type,
            sample_size=None
        )
        
        # Generate embeddings for validation data
        embeddings = temp_loader.embedder.generate_embeddings(
            data['premise_embeddings'],
            data['hypothesis_embeddings'],
            batch_size=1020
        )
        
        return embeddings, data['labels']


    def extract_latent_features_by_class(self, model, embeddings, labels, max_samples_per_class=1000):
        """Extract latent features from your best model, organized by class"""
        latent_features_by_class = defaultdict(list)
        
        # Handle both string and numeric labels
        if isinstance(labels[0], str):
            # Convert string labels to numeric
            label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            labels_numeric = [label_to_idx[label] for label in labels]
            labels_to_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        else:
            # Already numeric
            labels_numeric = labels
            labels_to_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        print("Extracting latent features from best model...")
        print(f"Label format: {type(labels[0])}")
        print(f"Sample labels: {labels[:5]}")
        
        # Convert to tensors
        embeddings_tensor = embeddings.to(self.device)
        labels_tensor = torch.tensor(labels_numeric).to(self.device)
        
        # Process in batches to avoid memory issues
        batch_size = 100
        total_samples = len(embeddings_tensor)
        
        with torch.no_grad():
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_embeddings = embeddings_tensor[start_idx:end_idx]
                batch_labels = labels_tensor[start_idx:end_idx]
                
                # Get latent representations
                latent_features = model.encoder(batch_embeddings)
                
                # Group by class
                for i in range(len(batch_labels)):
                    class_name = labels_to_names[batch_labels[i].item()]
                    if len(latent_features_by_class[class_name]) < max_samples_per_class:
                        latent_features_by_class[class_name].append(
                            latent_features[i].cpu().numpy()
                        )
                
                # Stop when we have enough samples
                if all(len(features) >= max_samples_per_class 
                       for features in latent_features_by_class.values()):
                    break
                    
                if start_idx % (batch_size * 50) == 0:
                    print(f"  Processed {start_idx}/{total_samples} samples...")
        
        # Convert to numpy arrays
        for class_name in latent_features_by_class:
            latent_features_by_class[class_name] = np.array(latent_features_by_class[class_name])
            print(f"  {class_name}: {latent_features_by_class[class_name].shape[0]} samples")
            
        return latent_features_by_class
    
    def compute_class_persistence_diagrams(self, latent_features_by_class):
        """Compute persistence diagrams using the CORRECT method from the project"""
        class_diagrams = {}
        
        print("\nComputing persistence diagrams for each class...")
        
        for class_name, features in latent_features_by_class.items():
            print(f"  Computing for {class_name}...")
            
            try:
                # Compute distance matrix using cosine distance (as used in project)
                distance_matrix = pairwise_distances(features, metric='cosine')
                
                # Use the CORRECT method from the project
                phd_score, diagrams = ph_dim_and_diagrams_from_distance_matrix(
                    distance_matrix,
                    min_points=200,
                    max_points=min(1000, len(features)),  # Don't exceed available samples
                    h_dim=1
                )
                
                h1_diagram = diagrams[1]  # Extract H1 features
                
                # Filter finite values only
                finite_mask = np.isfinite(h1_diagram).all(axis=1)
                h1_diagram = h1_diagram[finite_mask]
                
                class_diagrams[class_name] = h1_diagram
                print(f"    {class_name}: {len(h1_diagram)} H1 features")
                print(f"    PH dimension: {phd_score:.4f}")
                
            except Exception as e:
                print(f"    Error computing {class_name}: {e}")
                class_diagrams[class_name] = np.array([]).reshape(0, 2)
        
        return class_diagrams

    def load_prototypes(self):
        """Load your prototype diagrams"""
        if not self.prototype_path:
            print("No prototype path provided")
            return None
            
        try:
            with open(self.prototype_path, 'rb') as f:
                prototypes = pickle.load(f)
            print(f"Loaded prototypes from {self.prototype_path}")
            return prototypes
        except Exception as e:
            print(f"Error loading prototypes: {e}")
            return None
    
    def compare_with_prototypes(self, current_diagrams, prototypes):
        """Compare current diagrams with prototypes"""
        if prototypes is None:
            print("No prototypes to compare with")
            return
            
        print("\n" + "="*60)
        print("COMPARISON: Current Model vs Target Prototypes")
        print("="*60)
        
        overall_gaps = {}
        
        for class_name in current_diagrams:
            print(f"\n{class_name.upper()}:")
            
            current = current_diagrams[class_name]
            
            # Get prototype (adjust key structure based on your prototype format)
            if class_name in prototypes and 'H1' in prototypes[class_name]:
                target = prototypes[class_name]['H1']
            else:
                print(f"  No prototype found for {class_name}")
                continue
            
            # Basic statistics comparison
            print(f"  Current model:")
            print(f"    H1 features: {len(current)}")
            if len(current) > 0:
                persistences = current[:, 1] - current[:, 0]
                current_total = np.sum(persistences)
                current_max = np.max(persistences)
                current_mean = np.mean(persistences)
                print(f"    Total persistence: {current_total:.4f}")
                print(f"    Max persistence: {current_max:.4f}")
                print(f"    Mean persistence: {current_mean:.4f}")
            else:
                current_total = 0
                print(f"    No H1 features detected!")
            
            print(f"  Target prototype:")
            print(f"    H1 features: {len(target)}")
            if len(target) > 0:
                target_persistences = target[:, 1] - target[:, 0]
                target_total = np.sum(target_persistences)
                target_max = np.max(target_persistences)
                target_mean = np.mean(target_persistences)
                print(f"    Total persistence: {target_total:.4f}")
                print(f"    Max persistence: {target_max:.4f}")
                print(f"    Mean persistence: {target_mean:.4f}")
            else:
                target_total = 0
            
            # Gap analysis
            if current_total > 0 and target_total > 0:
                gap_ratio = current_total / target_total
                overall_gaps[class_name] = gap_ratio
                print(f"  Gap ratio (current/target): {gap_ratio:.4f}")
                
                if gap_ratio < 0.5:
                    print(f"  ❌ Current model has much LOWER complexity than target")
                elif gap_ratio > 2.0:
                    print(f"  ❌ Current model has much HIGHER complexity than target")
                else:
                    print(f"  ✅ Current model is reasonably close to target")
            elif current_total == 0:
                print(f"  ❌ Current model has NO topological complexity!")
                overall_gaps[class_name] = 0
            elif target_total == 0:
                print(f"  ❌ Target has no complexity (unexpected)")
                overall_gaps[class_name] = float('inf')
        
        # Overall assessment
        print(f"\n" + "="*60)
        print("OVERALL ASSESSMENT:")
        print("="*60)
        
        if all(gap == 0 for gap in overall_gaps.values()):
            print("❌ FUNDAMENTAL PROBLEM: Current model has NO topological complexity")
            print("   The latent space is too simple/uniform to generate H1 features")
            print("   Recommendation: Need to establish basic topological structure first")
        elif all(0.5 <= gap <= 2.0 for gap in overall_gaps.values() if gap > 0):
            print("✅ GOOD NEWS: Current model is reasonably close to targets")
            print("   Small-scale topological regularization should work")
        else:
            print("⚠️  MIXED RESULTS: Some classes close, others far off")
            print("   May need class-specific regularization weights")
            
        return overall_gaps
    
    def visualize_comparison(self, current_diagrams, prototypes=None):
        """Create visualization comparing current vs prototype diagrams"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        class_names = ['entailment', 'neutral', 'contradiction']
        colors = ['green', 'blue', 'red']
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            # Current model diagrams (top row)
            ax_current = axes[0, i]
            if class_name in current_diagrams:
                current = current_diagrams[class_name]
                if len(current) > 0:
                    ax_current.scatter(current[:, 0], current[:, 1], 
                                     alpha=0.7, color=color, s=50)
                    max_val = max(current.flatten()) if len(current) > 0 else 1
                    ax_current.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                else:
                    ax_current.text(0.5, 0.5, 'No H1 Features', 
                                  transform=ax_current.transAxes, ha='center', va='center')
            
            ax_current.set_title(f'{class_name.title()} - Current Model')
            ax_current.set_xlabel('Birth')
            ax_current.set_ylabel('Death')
            ax_current.grid(True, alpha=0.3)
            
            # Prototype diagrams (bottom row)
            ax_proto = axes[1, i]
            if prototypes and class_name in prototypes and 'H1' in prototypes[class_name]:
                target = prototypes[class_name]['H1']
                if len(target) > 0:
                    ax_proto.scatter(target[:, 0], target[:, 1], 
                                   alpha=0.7, color=color, s=50)
                    max_val = max(target.flatten()) if len(target) > 0 else 1
                    ax_proto.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                else:
                    ax_proto.text(0.5, 0.5, 'No H1 Features', 
                                transform=ax_proto.transAxes, ha='center', va='center')
            else:
                ax_proto.text(0.5, 0.5, 'No Prototype', 
                            transform=ax_proto.transAxes, ha='center', va='center')
            
            ax_proto.set_title(f'{class_name.title()} - Target Prototype')
            ax_proto.set_xlabel('Birth')
            ax_proto.set_ylabel('Death')
            ax_proto.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('persistence_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Visualization saved as 'persistence_comparison.png'")
    
    def run_full_diagnostic(self):
        """Run the complete diagnostic analysis"""
        print("="*60)
        print("DIAGNOSTIC ANALYSIS: Best Model vs Target Prototypes")
        print("="*60)
        
        # Load model
        model = self.load_model()
        
        # Load validation data
        embeddings, labels = self.load_validation_data()
        
        # Extract features by class
        latent_features = self.extract_latent_features_by_class(model, embeddings, labels)
        
        # Compute current persistence diagrams using CORRECT method
        current_diagrams = self.compute_class_persistence_diagrams(latent_features)
        
        # Load prototypes
        prototypes = self.load_prototypes()
        
        # Compare
        gaps = self.compare_with_prototypes(current_diagrams, prototypes)
        
        # Visualize
        self.visualize_comparison(current_diagrams, prototypes)
        
        return current_diagrams, prototypes, gaps

# Usage example:
if __name__ == "__main__":
    # Set up paths (adjust these to your actual paths)
    model_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/attention_topological_autoencoder_torchph_phase1_20250722_141516/checkpoints/best_model.pt"
    
    data_paths = {
        'train': "data/processed/snli_full_standard_SBERT.pt",
        'val': "data/processed/snli_full_standard_SBERT_validation.pt", 
        'test': "data/processed/snli_full_standard_SBERT_test.pt"
    }
    
    prototype_path = "entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototypes_medoid.pkl"
    
    # Run diagnostic
    diagnostic = BestModelDiagnostic(
        model_path=model_path,
        data_paths=data_paths,
        prototype_path=prototype_path,
        embedding_type='concat'  # Adjust based on your best model
    )
    
    current_diagrams, prototypes, gaps = diagnostic.run_full_diagnostic()
    
    # Print summary
    print(f"\nSUMMARY:")
    for class_name, gap in gaps.items():
        if gap == 0:
            print(f"{class_name}: No topological complexity")
        elif gap < 0.5:
            print(f"{class_name}: Much simpler than target ({gap:.2f}x)")
        elif gap > 2.0:
            print(f"{class_name}: Much more complex than target ({gap:.2f}x)")
        else:
            print(f"{class_name}: Close to target ({gap:.2f}x)")

    


    