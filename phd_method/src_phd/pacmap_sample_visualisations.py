"""
Sample-Level PaCMAP Visualization for SNLI Persistence Images

Creates PaCMAP visualization of individual premise-hypothesis pair topological signatures
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pacmap
from pathlib import Path
from sklearn.utils import resample
import torch

class SampleLevelPaCMAPVisualizer:
    """
    Create PaCMAP visualizations of individual sample persistence images
    """
    
    def __init__(self, chunk_number=1, random_state=42):
        self.chunk_number = chunk_number
        self.random_state = random_state
        
        # Same color scheme as class-level analysis
        self.class_colors = {
            0: '#27AE60',  # entailment - green
            1: '#F39C12',  # neutral - orange  
            2: '#E74C3C'   # contradiction - red
        }
        self.class_names = ['entailment', 'neutral', 'contradiction']
        
        # PaCMAP parameters - let it auto-select n_neighbors for ~100k samples
        self.pacmap_reducer = pacmap.PaCMAP(
            n_neighbors=None,  # Auto-select optimal value
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=random_state
        )
    
    def load_single_chunk_data(self):
        """
        Load a single chunk of SNLI training persistence images (~100k samples)
        """
        chunk_path = f"/vol/bitbucket/ahb24/tda_entailment_new/chunked_snli_train_persistence_images_chunk_{self.chunk_number}_of_5.pkl"
        
        print(f"Loading SNLI chunk {self.chunk_number}: {chunk_path}")
        
        if not Path(chunk_path).exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
        
        images = chunk_data['persistence_images']
        labels = chunk_data['labels']
        
        # Convert string labels to indices if needed
        label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        if isinstance(labels[0], str):
            labels = [label_to_idx[label] for label in labels]
        
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} samples from chunk {self.chunk_number}")
        
        # Print class distribution
        for class_idx in range(3):
            class_count = np.sum(labels == class_idx)
            class_pct = class_count / len(labels) * 100
            print(f"  {self.class_names[class_idx]}: {class_count:,} samples ({class_pct:.1f}%)")
        
        return images, labels
    
    def load_validation_data(self, max_samples=5000):
        """
        Load validation data for comparison
        """
        print("Loading SNLI validation data...")
        
        val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_persistence_images.pkl"
        
        if not Path(val_path).exists():
            print("Validation file not found, skipping...")
            return None, None
        
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        
        images = val_data['persistence_images']
        labels = val_data['labels']
        
        # Convert string labels to indices if needed
        label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        if isinstance(labels[0], str):
            labels = np.array([label_to_idx[label] for label in labels])
        else:
            labels = np.array(labels)
        
        # Sample if too large
        if len(images) > max_samples:
            images, labels = resample(
                images, labels,
                n_samples=max_samples,
                random_state=self.random_state,
                stratify=labels
            )
        
        print(f"  Loaded {len(images)} validation samples")
        return images, labels
    
    def create_pacmap_visualization(self, persistence_images, labels, 
                                  title="Sample-Level Topological Signatures", 
                                  save_path=None, alpha=0.6):
        """
        Create PaCMAP visualization of individual samples
        """
        print(f"Creating PaCMAP visualization for {len(persistence_images)} samples...")
        
        # Flatten persistence images if needed (30x30 -> 900D)
        if len(persistence_images.shape) == 3:
            X = persistence_images.reshape(len(persistence_images), -1)
        else:
            X = persistence_images
        
        print(f"  Input shape: {X.shape}")
        print(f"  Computing PaCMAP embedding...")
        
        # Apply PaCMAP
        X_embedded = self.pacmap_reducer.fit_transform(X)
        
        print(f"  Creating visualization...")
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot each class
        for class_idx in range(3):
            mask = labels == class_idx
            n_points = np.sum(mask)
            
            if n_points > 0:
                plt.scatter(
                    X_embedded[mask, 0],
                    X_embedded[mask, 1],
                    c=self.class_colors[class_idx],
                    label=f"{self.class_names[class_idx].title()} (n={n_points})",
                    alpha=alpha,
                    s=20,  # Smaller points for individual samples
                    edgecolors='none'
                )
        
        plt.xlabel("PaCMAP Component 1", fontsize=14)
        plt.ylabel("PaCMAP Component 2", fontsize=14)
        plt.title(title, fontsize=16, pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add sample info
        total_samples = len(persistence_images)
        info_text = f"Individual premise-hypothesis topological signatures (n={total_samples:,})"
        plt.figtext(0.02, 0.02, info_text, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Visualization saved to: {save_path}")
        
        plt.show()
        
        return X_embedded
    
    def run_single_chunk_analysis(self, save_dir="phd_method/sample_level_pacmap"):
        """
        Run PaCMAP analysis on a single chunk of ~100k samples
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print(f"SINGLE CHUNK SAMPLE-LEVEL ANALYSIS (Chunk {self.chunk_number})")
        print("="*60)
        
        # Load chunk data
        images, labels = self.load_single_chunk_data()
        
        # Create PaCMAP visualization
        title = f"Sample-Level Topological Signatures\nSNLI Training Chunk {self.chunk_number} (~{len(images):,} samples)"
        save_path = f"{save_dir}/sample_level_pacmap_chunk_{self.chunk_number}.png"
        
        embedding = self.create_pacmap_visualization(
            images, labels,
            title=title,
            save_path=save_path,
            alpha=0.3  # Lower alpha for better visibility with many points
        )
        
        print("\n" + "="*60)
        print("SINGLE CHUNK ANALYSIS COMPLETE")
        print("="*60)
        print(f"Processed {len(images):,} individual samples")
        print("This shows the topological distribution of individual premise-hypothesis pairs")
        print("vs the class-level aggregated signatures from your earlier analysis.")
        
        return embedding, images, labels


def main():
    """
    Run sample-level PaCMAP analysis on a single chunk
    """
    
    print("Sample-Level Topological PaCMAP Analysis")
    print("Single Chunk Analysis (~100k samples)")
    print("="*50)
    
    # Create visualizer for chunk 1 (you can change this to any chunk 1-5)
    visualizer = SampleLevelPaCMAPVisualizer(
        chunk_number=1,  # Use chunk 1
        random_state=42
    )
    
    # Run analysis on single chunk
    embedding, images, labels = visualizer.run_single_chunk_analysis()
    
    print(f"\nSample-level PaCMAP analysis completed!")
    print(f"Analyzed {len(images):,} individual persistence images")
    print("This visualization shows individual sample topology vs your class-level analysis")
    print("which showed aggregated topological signatures.")


if __name__ == "__main__":
    main()