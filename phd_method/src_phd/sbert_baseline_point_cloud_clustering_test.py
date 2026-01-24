"""
Point Cloud Clustering Test - BASELINE VERSION (SBERT only, no learned models)
Tests clustering accuracy using only SBERT tokens without Order/Asymmetry transformations
"""

import os
import sys
import json
import numpy as np
import torch
import random
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from persim import PersistenceImager
from itertools import permutations
import matplotlib.pyplot as plt
from gph.python import ripser_parallel
from sklearn.metrics.pairwise import pairwise_distances

# For classification:
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# For topological nn classifier:
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For feature importance analysis:
from scipy.stats import f_oneway
import pandas as pd
from tqdm import tqdm


@dataclass
class ClusteringResult:
    """Results for point cloud clustering test"""
    model_name: str
    clustering_accuracy: float
    silhouette_score: float
    adjusted_rand_score: float
    num_samples: int
    success: bool
    ph_dim_values: Dict[str, List[float]]
    ph_dim_stats: Dict[str, Dict[str, float]]
    point_cloud_stats: Dict[str, Dict[str, float]]


class BaselinePointCloudGenerator:
    """Generate point clouds using ONLY SBERT tokens (no learned transformations)"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Baseline Point Cloud Generator initialized on {self.device}")
        print("Using ONLY SBERT tokens - no Order or Asymmetry models")
    
    def generate_point_cloud_variations(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate point cloud variations using ONLY original SBERT tokens
        
        Args:
            tokens: SBERT token embeddings [num_tokens, 768]
            
        Returns:
            List with single point cloud: [original_tokens]
        """
        point_clouds = []
        
        with torch.no_grad():
            tokens = tokens.to(self.device)
            
            # Only original SBERT tokens
            point_clouds.append(tokens.cpu().clone())
        
        return point_clouds
    
    def generate_premise_hypothesis_point_cloud(self, premise_tokens: torch.Tensor, 
                                               hypothesis_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Generate combined point cloud from premise-hypothesis pair
        BASELINE: Only uses original SBERT tokens
        """
        # Generate premise point clouds (just SBERT tokens)
        premise_clouds = self.generate_point_cloud_variations(premise_tokens)
        
        # Generate hypothesis point clouds (just SBERT tokens)
        hypothesis_clouds = self.generate_point_cloud_variations(hypothesis_tokens)
        
        # Combine all point clouds
        all_clouds = premise_clouds + hypothesis_clouds
        combined_cloud = torch.cat(all_clouds, dim=0)
        
        # Generate statistics
        stats = {
            'premise_points': premise_clouds[0].shape[0],
            'hypothesis_points': hypothesis_clouds[0].shape[0],
            'combined_total_points': combined_cloud.shape[0],
            'sufficient_for_phd': combined_cloud.shape[0] >= 20  # Lower threshold for baseline
        }
        
        return combined_cloud, stats


class BaselineClusteringValidator:
    """
    Validator for point cloud clustering using ONLY SBERT tokens
    """
    
    def __init__(self, 
                 val_data_path: str,
                 output_dir: str = "phd_method/baseline_clustering_results/",
                 seed: int = 42):
        
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load validation data
        print(f"Loading validation data: {val_data_path}")
        with open(val_data_path, 'rb') as f:
            self.val_data = pickle.load(f)
        
        print(f"Loaded {len(self.val_data['labels'])} validation samples")
        
        # Initialize baseline point cloud generator
        self.point_cloud_generator = BaselinePointCloudGenerator()
        
        # Clustering parameters
        self.samples_per_class = 100
        self.min_points_for_phd = 20  # Much lower threshold for SBERT-only (we only have ~28 tokens)
        
        print(f"Baseline clustering validator initialized")
        print(f"Samples per class: {self.samples_per_class}")
    
    def ph_dim_and_diagrams_from_distance_matrix(self, dm: np.ndarray,
                                 min_points: int = 50,
                                 max_points: int = 1000,
                                 point_jump: int = 25,
                                 h_dim: int = 0,
                                 alpha: float = 1.0) -> Tuple[float, List]:
        """Compute persistence on FULL point cloud"""
        assert dm.ndim == 2 and dm.shape[0] == dm.shape[1]
        
        print(f"Computing persistence on full {dm.shape[0]} point cloud...")
    
        # Compute persistence diagrams on FULL point cloud
        full_diagrams = ripser_parallel(dm, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        
        print(f"  H0 features: {len(full_diagrams[0])}")
        print(f"  H1 features: {len(full_diagrams[1])}")
        
        # For PH-dimension, subsample
        test_n = range(min_points, min(max_points, dm.shape[0]), point_jump)
        lengths = []
        
        for points_number in test_n:
            if points_number >= dm.shape[0]:
                break
                
            sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
            dist_matrix = dm[sample_indices, :][:, sample_indices]
            
            sub_diagrams = ripser_parallel(dist_matrix, maxdim=0, n_threads=-1, metric="precomputed")['dgms']
            
            d = sub_diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())
        
        if len(lengths) < 2:
            ph_dimension = 0.0
        else:
            lengths = np.array(lengths)
            x = np.log(np.array(list(test_n[:len(lengths)])))
            y = np.log(lengths)
            N = len(x)
            
            if N < 2:
                ph_dimension = 0.0
            else:
                m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
                ph_dimension = alpha / (1 - m) if m != 1 else 0.0
        
        return ph_dimension, full_diagrams
    
    def persistence_diagrams_to_images(self, all_diagrams: List, target_resolution: int = 30) -> List[np.ndarray]:
        """Convert persistence diagrams to standardized images"""
        
        # Analyze actual data ranges
        all_birth_times = []
        all_death_times = []
        all_lifespans = []
        valid_diagrams_count = 0
        
        print("Analyzing persistence diagram ranges...")
        
        for diagram_idx, diagrams in enumerate(all_diagrams):
            if diagrams is None:
                continue
            if isinstance(diagrams, (list, tuple)) and len(diagrams) == 0:
                continue
            if isinstance(diagrams, np.ndarray) and diagrams.size == 0:
                continue

            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    if diagram.ndim == 1:
                        continue
                    elif diagram.ndim == 2 and diagram.shape[1] >= 2:
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        
                        if len(finite_diagram) > 0:
                            all_birth_times.extend(finite_diagram[:, 0])
                            all_death_times.extend(finite_diagram[:, 1])
                            lifespans = finite_diagram[:, 1] - finite_diagram[:, 0]
                            all_lifespans.extend(lifespans)
                            valid_diagrams_count += 1
        
        print(f"Found {valid_diagrams_count} valid diagrams with {len(all_lifespans)} total features")
        
        if len(all_lifespans) == 0:
            print("No finite features found!")
            return []
        
        # Calculate data ranges
        min_birth = np.min(all_birth_times)
        max_birth = np.max(all_birth_times)
        min_life = np.min(all_lifespans)
        max_life = np.max(all_lifespans)
        
        birth_padding = max(0.01, (max_birth - min_birth) * 0.1)
        life_padding = max(0.001, (max_life - min_life) * 0.1)
        
        birth_range = (max(0, min_birth - birth_padding), max_birth + birth_padding)
        pers_range = (max(0.001, min_life - life_padding), max_life + life_padding)
        
        pixel_size = max(0.001, (pers_range[1] - pers_range[0]) / 138.9)
        sigma = max(0.001, (pers_range[1] - pers_range[0]) / 82.6)
        
        pimgr = PersistenceImager(
            pixel_size=pixel_size,
            birth_range=birth_range,
            pers_range=pers_range,
            kernel_params={'sigma': sigma}
        )
        
        persistence_images = []
        successful_conversions = 0
        
        for diagram_idx, diagrams in enumerate(all_diagrams):
            if diagrams is None:
                continue
            
            combined_image = np.zeros((target_resolution, target_resolution))
            has_content = False
            
            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    if diagram.ndim == 2 and diagram.shape[1] >= 2:
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        
                        if len(finite_diagram) > 0:
                            try:
                                img = pimgr.transform([finite_diagram])[0]
                                
                                if img.shape != (target_resolution, target_resolution):
                                    from scipy.ndimage import zoom
                                    zoom_factors = (target_resolution / img.shape[0], target_resolution / img.shape[1])
                                    img = zoom(img, zoom_factors)
                                
                                combined_image += img
                                has_content = True
                                
                            except Exception as e:
                                continue
            
            if has_content and combined_image.max() > 0:
                combined_image = combined_image / combined_image.max()
                persistence_images.append(combined_image.flatten())
                successful_conversions += 1
        
        print(f"\nPersistence image conversion: {successful_conversions}/{len(all_diagrams)} successful")
        
        return persistence_images

    def compute_distance_matrix(self, point_cloud: torch.Tensor, metric: str = 'braycurtis') -> np.ndarray:
        """Compute distance matrix for point cloud"""
        point_cloud_np = point_cloud.numpy()
        distance_matrix = pairwise_distances(point_cloud_np, metric=metric)
        return distance_matrix

    def filter_samples_by_token_count(self, samples: List[Dict], min_combined_tokens: int = 0) -> List[Dict]:
        """Filter samples to ensure sufficient tokens"""
        filtered_samples = []
        
        for sample in samples:
            premise_tokens = sample['premise_tokens'].shape[0]
            hypothesis_tokens = sample['hypothesis_tokens'].shape[0]
            combined_tokens = premise_tokens + hypothesis_tokens
            
            if combined_tokens >= min_combined_tokens:
                filtered_samples.append(sample)
        
        print(f"Token filtering: {len(filtered_samples)}/{len(samples)} samples have ≥{min_combined_tokens} tokens")
        
        return filtered_samples

    def generate_maximum_samples_by_class(self) -> Dict[str, List[Dict]]:
        """Generate maximum available samples for each class"""
        
        print("Generating maximum sample indices...")
        
        class_data = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for i, label in enumerate(self.val_data['labels']):
            class_data[label].append({
                'index': i,
                'premise_tokens': self.val_data['premise_tokens'][i],
                'hypothesis_tokens': self.val_data['hypothesis_tokens'][i],
                'label': label
            })
        
        max_samples = {}
        
        for class_name, class_samples in class_data.items():
            print(f"  {class_name}: {len(class_samples)} total samples available")

            filtered_samples = self.filter_samples_by_token_count(class_samples)
            max_samples[class_name] = filtered_samples
            
            print(f"    Using ALL {len(filtered_samples)} samples after token filtering")

            token_counts = [
                s['premise_tokens'].shape[0] + s['hypothesis_tokens'].shape[0] 
                for s in filtered_samples
            ]
            print(f"Token count stats: {np.mean(token_counts):.0f} ± {np.std(token_counts):.0f} "
                f"(range: {np.min(token_counts)}-{np.max(token_counts)})")
        
        return max_samples

    def validate_baseline_clustering(self) -> ClusteringResult:
        """Main validation function - test point cloud clustering with SBERT only"""
        
        print("\n" + "="*80)
        print("BASELINE POINT CLOUD CLUSTERING VALIDATION (SBERT ONLY)")
        print("="*80)
        
        max_samples = self.generate_maximum_samples_by_class()
            
        # Initialize collection variables
        all_persistence_diagrams = []
        sample_labels = []
        ph_dim_values = {'entailment': [], 'neutral': [], 'contradiction': []}
        point_cloud_stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        # Process each class
        for class_idx, class_name in enumerate(['entailment', 'neutral', 'contradiction']):
            print(f"\nProcessing {class_name} samples...")
            
            class_samples = max_samples[class_name]

            for sample_idx, sample_data in enumerate(class_samples):
                premise_tokens = sample_data['premise_tokens']
                hypothesis_tokens = sample_data['hypothesis_tokens']
                
                # Generate point cloud using SBERT only
                point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                    premise_tokens, hypothesis_tokens
                )
                
                # Record statistics
                point_cloud_stats[class_name].append(stats)
                
                if sample_idx % 100 == 0:
                    print(f"  Sample {sample_idx}: {stats['combined_total_points']} points")
                
                # Skip if insufficient points
                if not stats['sufficient_for_phd']:
                    continue
                
                # Compute distance matrix
                distance_matrix = self.compute_distance_matrix(point_cloud)
                
                # Compute PHD and persistence diagrams
                ph_dim, diagrams = self.ph_dim_and_diagrams_from_distance_matrix(
                    distance_matrix,
                    min_points=50,
                    max_points=min(200, point_cloud.shape[0]),
                    point_jump=25
                )
                
                ph_dim_values[class_name].append(ph_dim)
                all_persistence_diagrams.append(diagrams)
                sample_labels.append(class_idx)

        print(f"\nCollected {len(all_persistence_diagrams)} diagram sets for clustering")

        # Convert diagrams to persistence images
        persistence_images = self.persistence_diagrams_to_images(all_persistence_diagrams)
                
        print(f"Generated {len(persistence_images)} persistence images for clustering")
        
        # Perform clustering analysis
        if len(persistence_images) > 0:
            print("Performing clustering analysis...")
            accuracy, sil_score, ari_score = self.perform_clustering_analysis(
                persistence_images, sample_labels
            )
        else:
            print("No persistence images generated - cannot perform clustering")
            accuracy, sil_score, ari_score = 0.0, 0.0, 0.0
        
        # Calculate statistics
        ph_dim_stats = {}
        comprehensive_point_stats = {}
        
        for class_name in ['entailment', 'neutral', 'contradiction']:
            ph_dims = ph_dim_values[class_name]
            if ph_dims:
                ph_dim_stats[class_name] = {
                    'mean': float(np.mean(ph_dims)),
                    'std': float(np.std(ph_dims)),
                    'min': float(np.min(ph_dims)),
                    'max': float(np.max(ph_dims))
                }
            
            class_stats = point_cloud_stats[class_name]
            if class_stats:
                combined_points = [s['combined_total_points'] for s in class_stats]
                premise_points = [s['premise_points'] for s in class_stats]
                hypothesis_points = [s['hypothesis_points'] for s in class_stats]
                
                comprehensive_point_stats[class_name] = {
                    'combined_mean': float(np.mean(combined_points)),
                    'combined_std': float(np.std(combined_points)),
                    'premise_mean': float(np.mean(premise_points)),
                    'hypothesis_mean': float(np.mean(hypothesis_points)),
                    'sufficient_rate': float(np.mean([s['sufficient_for_phd'] for s in class_stats]))
                }
        
        # Create result
        result = ClusteringResult(
            model_name="SBERT_baseline_only",
            clustering_accuracy=accuracy,
            silhouette_score=sil_score,
            adjusted_rand_score=ari_score,
            num_samples=sum(len(samples) for samples in max_samples.values()),
            success=(accuracy > 0.7),
            ph_dim_values=ph_dim_values,
            ph_dim_stats=ph_dim_stats,
            point_cloud_stats=comprehensive_point_stats
        )
        
        # Print results
        print(f"\n" + "="*80)
        print("BASELINE CLUSTERING RESULTS (SBERT ONLY)")
        print("="*80)
        print(f"Clustering Accuracy: {accuracy:.3f}")
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        print(f"Success (>70%): {'🎉 YES' if result.success else '❌ NO'}")
        
        print(f"\nPH-Dimension Statistics:")
        for class_name, stats in ph_dim_stats.items():
            print(f"  {class_name}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        print(f"\nPoint Cloud Statistics:")
        for class_name, stats in comprehensive_point_stats.items():
            print(f"  {class_name}:")
            print(f"    Combined: {stats['combined_mean']:.0f} ± {stats['combined_std']:.0f} points")
        
        return result

    def perform_clustering_analysis(self, persistence_images: List[np.ndarray], 
                               sample_labels: List[int]) -> Tuple[float, float, float]:
        """Perform clustering analysis on persistence images"""
        
        if len(persistence_images) == 0:
            print("No persistence images available for clustering")
            return 0.0, 0.0, 0.0
        
        print(f"Clustering {len(persistence_images)} persistence images...")
        
        X = np.array(persistence_images)
        y_true = np.array(sample_labels)
        
        n_clusters = len(np.unique(y_true))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
        # Find best label permutation
        best_accuracy = 0.0
        best_permutation = None
        
        for perm in permutations(range(n_clusters)):
            mapped_pred = np.array([perm[label] for label in y_pred])
            accuracy = np.mean(mapped_pred == y_true)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_permutation = perm
        
        final_pred = np.array([best_permutation[label] for label in y_pred])
        
        try:
            silhouette = silhouette_score(X, final_pred)
        except:
            silhouette = 0.0
        
        try:
            ari = adjusted_rand_score(y_true, final_pred)
        except:
            ari = 0.0
        
        print(f"Clustering results:")
        print(f"  Best accuracy: {best_accuracy:.3f}")
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"  Adjusted Rand Index: {ari:.3f}")
        
        return best_accuracy, silhouette, ari


def main():
    """Run baseline point cloud clustering validation"""
    
    val_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_test_sbert_tokens.pkl"
    output_dir = "MSc_Topology_Codebase/phd_method/baseline_clustering_results/"
    os.makedirs(output_dir, exist_ok=True)
    
    if not Path(val_data_path).exists():
        print(f"Validation data not found at: {val_data_path}")
        return
    
    # Run validation
    validator = BaselineClusteringValidator(
        val_data_path=val_data_path,
        output_dir=output_dir,
        seed=42
    )
    
    result = validator.validate_baseline_clustering()
    
    print("\n" + "="*80)
    print("BASELINE CLUSTERING VALIDATION COMPLETED!")
    
    if result.success:
        print("🎉 SUCCESS: Achieved >70% clustering accuracy with SBERT only!")
        print("This suggests topology exists in the raw embeddings.")
    else:
        print("❌ Did not achieve 70% clustering threshold")
        print("This suggests learned transformations are necessary.")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()