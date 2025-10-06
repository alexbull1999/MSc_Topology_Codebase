"""
PaCMAP Visualization of Topological Entailment Signatures

Clean implementation that uses the correct functions from phdim_clustering.py
and adds PaCMAP visualization for the persistence images that achieved 100% clustering.
Uses Euclidean distance for n_neighbours selection inside algorithm, 
but bray-curtis/cosine on distance matrices for persistence image creation
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pacmap
from sklearn.metrics import pairwise_distances
from persim import PersistenceImager
from gph.python import ripser_parallel
from scipy.ndimage import zoom
from itertools import permutations

# Add paths for existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phdim_distance_metric_optimized import SurfaceDistanceMetricAnalyzer


def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                            min_points=200,
                                            max_points=1000,
                                            point_jump=50,
                                            h_dim=0,
                                            alpha: float = 1.,
                                            seed: int = 42) -> Tuple[float, List]:
    """
    Modified version of ph_dim_from_distance_matrix that returns BOTH 
    PH-dimension and persistence diagrams
    
    Returns:
        Tuple of (ph_dimension, list_of_persistence_diagrams)
    """
    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape

    test_n = range(min_points, max_points, point_jump)
    lengths = []
    all_diagrams = []  # Store all persistence diagrams

    for points_number in test_n:
        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]

        # Compute persistence diagrams
        diagrams = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        all_diagrams.append(diagrams)  # Store the diagrams

        # Extract specific dimension for PH-dim calculation
        d = diagrams[h_dim]
        d = d[d[:, 1] < np.inf]
        lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())

    lengths = np.array(lengths)

    # Compute PH dimension
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    
    ph_dimension = alpha / (1 - m)
    
    return ph_dimension, all_diagrams


def persistence_diagrams_to_images(all_diagrams: List) -> List[np.ndarray]:
    """
    Convert persistence diagrams to standardized images
    Same function as in the clustering code
    """
    pimgr = PersistenceImager(
        pixel_size=0.5,
        birth_range=(0, 5),
        pers_range=(0, 5),
        kernel_params={'sigma': 0.3}
    )
    
    persistence_images = []
    
    for diagrams in all_diagrams:
        combined_image = np.zeros((20, 20))  # Fixed size
        
        # Process H0 and H1 diagrams
        for dim in range(min(2, len(diagrams))):
            diagram = diagrams[dim]
            if len(diagram) > 0:
                # Filter infinite persistence
                finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
                if len(finite_diagram) > 0:
                    try:
                        img = pimgr.transform([finite_diagram])[0]
                        # Resize to standard size if needed
                        if img.shape != (20, 20):
                            zoom_factors = (20 / img.shape[0], 20 / img.shape[1])
                            img = zoom(img, zoom_factors)
                        combined_image += img
                    except:
                        continue
        
        # Normalize and flatten
        if combined_image.max() > 0:
            combined_image = combined_image / combined_image.max()
        
        persistence_images.append(combined_image.flatten())
    
    return persistence_images


class PaCMAPTopologyVisualizer:
    """
    Visualize distinct class-based topology using PaCMAP on persistence images
    that achieved 100% clustering accuracy in the thesis
    """
    
    def __init__(self, 
                 bert_data_path: str,
                 order_model_path: str,
                 samples_per_class=10, 
                 points_per_sample=1000,
                 seed=42):
        
        self.samples_per_class = samples_per_class
        self.points_per_sample = points_per_sample
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.class_names = ['entailment', 'neutral', 'contradiction']
        
        # Updated color scheme with specified RGB values
        self.class_colors = {
            'entailment': (136/255, 177/255, 113/255),      # Green
            'neutral': (155/255, 90/255, 152/255),          # Purple
            'contradiction': (103/255, 137/255, 157/255)    # Blue
        }
        
        # Initialize PaCMAP with settings for topology preservation
        self.pacmap_reducer = pacmap.PaCMAP(
            n_neighbors=10,
            MN_ratio=0.5, 
            FP_ratio=2.0,
            random_state=seed
        )
        
        # Initialize analyzer using existing infrastructure
        print("Initializing SurfaceDistanceMetricAnalyzer...")
        self.analyzer = SurfaceDistanceMetricAnalyzer(
            bert_data_path=bert_data_path,
            order_model_path=order_model_path,
            seed=seed
        )
        
        # Target embedding spaces (the best performing ones from thesis)
        self.target_spaces = [
            'sbert_concat',
            'order_concat',
            'lattice_containment',
            'hyperbolic_concat'
        ]
        
        # Distance metrics that worked best
        self.distance_metrics = [
            'cosine',
            'braycurtis',
            'euclidean',
            'manhattan'
        ]
    
    def generate_fixed_samples_from_embeddings(self, all_embeddings):
        """
        Generate fixed sample indices that will be used across all embedding spaces
        Same as clustering code
        """
        print("Generating fixed sample indices...")
        
        fixed_sample_indices = {}
        
        # For each class, generate fixed sample indices
        for class_idx, class_name in enumerate(self.class_names):
            class_indices = []
            
            # Get the size from the first embedding space (they should all be the same)
            first_space = list(all_embeddings.keys())[0]
            n_total = all_embeddings[first_space][class_name].shape[0]
            
            print(f"  {class_name}: {n_total} total samples available")
            
            # Generate sample indices for this class
            base_seed = self.seed + class_idx * 100
            
            for sample_idx in range(self.samples_per_class):
                sample_seed = base_seed + sample_idx
                np.random.seed(sample_seed)
                
                if self.points_per_sample > n_total:
                    indices = np.arange(n_total)
                    actual_points = n_total
                else:
                    indices = np.random.choice(n_total, self.points_per_sample, replace=False)
                    actual_points = self.points_per_sample
                
                class_indices.append({
                    'indices': indices,
                    'actual_points': actual_points,
                    'sample_idx': sample_idx,
                    'class_name': class_name,
                    'class_idx': class_idx
                })
            
            fixed_sample_indices[class_name] = class_indices
            print(f"    Generated {len(class_indices)} sample index sets")
        
        return fixed_sample_indices
    
    def generate_persistence_images_for_space_metric(self, space_name, metric_name, 
                                                   all_embeddings, fixed_sample_indices):
        """
        Generate persistence images for one embedding space + metric combination
        Uses the exact same approach as the clustering validation
        """
        print(f"Generating persistence images for {space_name} + {metric_name}")
        
        try:
            space_embeddings = all_embeddings[space_name]
            
            all_persistence_images = []
            sample_labels = []
            
            # Process each fixed sample
            for class_name in self.class_names:
                class_embeddings = space_embeddings[class_name]
                class_sample_indices = fixed_sample_indices[class_name]
                class_idx = self.class_names.index(class_name)
                
                for sample_info in class_sample_indices:
                    indices = sample_info['indices']
                    
                    # Extract the sample using fixed indices
                    sampled_embeddings = class_embeddings[indices]
                    
                    # Compute distance matrix using existing analyzer method
                    distance_matrix = self.analyzer.compute_distance_matrix_advanced(
                        sampled_embeddings, metric_name
                    )
                    
                    # Get persistence diagrams
                    ph_dim, all_diagrams = ph_dim_and_diagrams_from_distance_matrix(
                        distance_matrix,
                        min_points=min(200, sampled_embeddings.shape[0]//2),
                        max_points=min(1000, sampled_embeddings.shape[0]),
                        point_jump=50,
                        h_dim=0,
                        alpha=1.0
                    )
                    

                    # Convert ALL persistence diagrams to images
                    for diagram in all_diagrams:
                        persistence_image = persistence_diagrams_to_images([diagram])
                        
                        if len(persistence_image) > 0:
                            all_persistence_images.append(persistence_image[0])
                            sample_labels.append(class_idx)
                print(f"  Generated {len([s for s in sample_labels if s == class_idx])} images for {class_name}")
            
            return all_persistence_images, sample_labels
            
        except Exception as e:
            print(f"  ERROR generating persistence images: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def visualize_with_pacmap(self, persistence_images, labels, 
                            space_name, metric_name, save_path=None):
        """
        Create PaCMAP visualization of persistence images
        """
        print(f"Creating PaCMAP visualization for {space_name} + {metric_name}...")

        if space_name == "sbert_concat":
            space_name = "Simple Concatenation"
        elif space_name == "lattice_containment":
            space_name = "Semantic Coherence"
        
        if metric_name == "braycurtis":
            metric_name = "Bray-Curtis"
        
        # Convert to numpy arrays
        X = np.array(persistence_images)
        y = np.array(labels)
        
        print(f"  Applying PaCMAP to {X.shape[0]} images of dimension {X.shape[1]}")
        
        # Apply PaCMAP embedding
        X_embedded = self.pacmap_reducer.fit_transform(X)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each class
        for class_idx, class_name in enumerate(self.class_names):
            mask = y == class_idx
            n_points = np.sum(mask)
            
            if n_points > 0:
                scatter = ax.scatter(
                    X_embedded[mask, 0], 
                    X_embedded[mask, 1],
                    c=self.class_colors[class_name],
                    label=f"{class_name.title()} (n={n_points})",
                    alpha=0.8,
                    s=80,
                    edgecolors='white',
                    linewidth=0.5
                )
        
        ax.set_xlabel("PaCMAP Component 1", fontsize=14)
        ax.set_ylabel("PaCMAP Component 2", fontsize=14)
        
        title = f"PaCMAP: Topological Signatures\n{space_name.replace('_', ' ').title()} + {metric_name.title()} Distance"
        ax.set_title(title, fontsize=16, pad=20)
        
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        
        # plt.figtext(0.02, 0.02, fontsize=10, 
        #            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
            # Save PDF for paper (editable)
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"PDF saved to {pdf_path}")
            
            # Optional: Also save SVG
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"SVG saved to {svg_path}")
        
        plt.show()
        
        return X_embedded
    
    def run_visualization_pipeline(self, target_spaces=None, target_metrics=None, 
                                 save_dir='entailment_surfaces/pacmap_visualizations'):
        """
        Run the complete pipeline for PaCMAP visualizations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("="*80)
        print("PaCMAP TOPOLOGICAL VISUALIZATION PIPELINE")
        print("="*80)
        print(f"Samples per Class: {self.samples_per_class}")
        print(f"Points per Sample: {self.points_per_sample}")
        print("="*80)
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Use defaults if not specified
        if target_spaces is None:
            target_spaces = self.target_spaces
        if target_metrics is None:
            target_metrics = self.distance_metrics
        
        try:
            # Step 1: Generate all embedding spaces
            print("Step 1: Extracting embedding spaces...")
            all_embeddings = self.analyzer.extract_all_embedding_spaces(max_samples_per_class=None)
            
            # Filter to target spaces
            target_embeddings = {}
            for space_name in target_spaces:
                if space_name in all_embeddings:
                    target_embeddings[space_name] = all_embeddings[space_name]
                    print(f"  {space_name}: Available")
                else:
                    print(f"  {space_name}: Not available")
            
            if not target_embeddings:
                print("ERROR: No target embedding spaces available!")
                return
            
            # Step 2: Generate fixed sample indices
            print("\nStep 2: Generating fixed sample indices...")
            fixed_sample_indices = self.generate_fixed_samples_from_embeddings(target_embeddings)
            
            # Step 3: Generate visualizations for each space + metric combination
            print("\nStep 3: Generating PaCMAP visualizations...")
            
            results = {}
            
            for space_name in target_embeddings.keys():
                for metric_name in target_metrics:
                    print(f"\n{'='*60}")
                    print(f"PROCESSING: {space_name} + {metric_name}")
                    print(f"{'='*60}")
                    
                    # Generate persistence images
                    persistence_images, labels = self.generate_persistence_images_for_space_metric(
                        space_name, metric_name, target_embeddings, fixed_sample_indices
                    )
                    
                    if len(persistence_images) > 0:
                        # Create PaCMAP visualization
                        save_path = f"{save_dir}/SNLI_pacmap_{space_name}_{metric_name}_{timestamp}.png"
                        embedding_2d = self.visualize_with_pacmap(
                            persistence_images, labels, space_name, metric_name, save_path
                        )
                        
                        # Store results
                        results[f"{space_name}_{metric_name}"] = {
                            'space_name': space_name,
                            'metric_name': metric_name,
                            'n_images': len(persistence_images),
                            'embedding_2d': embedding_2d.tolist(),
                            'labels': labels,
                            'save_path': save_path
                        }
                        
                        print(f"SUCCESS: {space_name} + {metric_name}")
                    else:
                        print(f"FAILED: {space_name} + {metric_name} - No persistence images generated")
            
            # Save all results
            results_path = f"{save_dir}/all_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for key, val in results.items():
                    json_results[key] = {
                        'space_name': val['space_name'],
                        'metric_name': val['metric_name'], 
                        'n_images': val['n_images'],
                        'labels': val['labels'],
                        'save_path': val['save_path']
                        # Skip embedding_2d for JSON as it's large
                    }
                json.dump(json_results, f, indent=2)
            
            print("="*80)
            print("PIPELINE COMPLETED")
            print("="*80)
            print(f"Generated {len(results)} visualizations")
            print(f"Results saved: {results_path}")
            
            return results
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """
    Run PaCMAP visualizations for the best performing embedding spaces
    """
    
    # These paths need to be updated to your actual file locations
    bert_data_path = "data/processed/snli_full_standard_SBERT.pt"
    order_model_path = "models/enhanced_order_embeddings_snli_SBERT_full.pt"
    
    visualizer = PaCMAPTopologyVisualizer(
        bert_data_path=bert_data_path,
        order_model_path=order_model_path,
        samples_per_class=100, 
        points_per_sample=1000,
        seed=42
    )
    
    # Test the best performing combinations from the thesis
    best_spaces = ['sbert_concat', 'lattice_containment']  # Start with these
    best_metrics = ['cosine', 'braycurtis']  # These worked well
    # best_metrics = ['euclidean']
    
    results = visualizer.run_visualization_pipeline(
        target_spaces=best_spaces,
        target_metrics=best_metrics
    )
    
    print(f"\nGenerated {len(results)} PaCMAP visualizations showing distinct topological signatures!")


if __name__ == "__main__":
    main()