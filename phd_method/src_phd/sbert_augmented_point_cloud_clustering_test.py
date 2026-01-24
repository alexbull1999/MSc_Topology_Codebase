"""
Point Cloud Clustering Test - AUGMENTED BASELINE
SBERT tokens + geometric augmentation (NO learned transformations)
Tests whether improvement comes from learned models or just more points
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


class AugmentedBaselineGenerator:
    """
    Generate point clouds using SBERT tokens + geometric augmentation
    NO learned transformations - just interpolation, jittering, etc.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Augmented Baseline Generator initialized on {self.device}")
        print("Using SBERT tokens + geometric augmentation (NO learned models)")
    
    def generate_interpolated_points(self, tokens: torch.Tensor, num_interpolations: int = 50) -> torch.Tensor:
        """
        Generate interpolated points between token pairs
        Simple geometric augmentation without learned transformations
        """
        if tokens.shape[0] < 2:
            return torch.empty(0, tokens.shape[1])
        
        interpolated = []
        num_tokens = tokens.shape[0]
        
        # Pairwise interpolation
        for i in range(min(num_interpolations, num_tokens * (num_tokens - 1) // 2)):
            # Random pair selection
            idx1, idx2 = np.random.choice(num_tokens, 2, replace=False)
            
            # Random interpolation weight
            alpha = np.random.uniform(0.2, 0.8)
            
            # Linear interpolation
            interp_point = alpha * tokens[idx1] + (1 - alpha) * tokens[idx2]
            interpolated.append(interp_point)
        
        if interpolated:
            return torch.stack(interpolated)
        return torch.empty(0, tokens.shape[1])
    
    def generate_jittered_points(self, tokens: torch.Tensor, num_jitter: int = 30, noise_scale: float = 0.1) -> torch.Tensor:
        """
        Add Gaussian noise to tokens to increase point cloud density
        """
        if tokens.shape[0] == 0:
            return torch.empty(0, tokens.shape[1])
        
        jittered = []
        
        for _ in range(num_jitter):
            idx = np.random.choice(tokens.shape[0])
            noise = torch.randn_like(tokens[idx]) * noise_scale
            jittered_point = tokens[idx] + noise
            jittered.append(jittered_point)
        
        return torch.stack(jittered)
    
    def generate_centroid_variants(self, tokens: torch.Tensor, num_variants: int = 20) -> torch.Tensor:
        """
        Generate points around the centroid at different distances
        """
        if tokens.shape[0] == 0:
            return torch.empty(0, tokens.shape[1])
        
        centroid = tokens.mean(dim=0)
        variants = []
        
        for _ in range(num_variants):
            # Random distance scaling
            scale = np.random.uniform(0.5, 1.5)
            
            # Random direction from centroid
            idx = np.random.choice(tokens.shape[0])
            direction = tokens[idx] - centroid
            
            variant = centroid + scale * direction
            variants.append(variant)
        
        return torch.stack(variants)
    
    def generate_premise_hypothesis_point_cloud(self, premise_tokens: torch.Tensor, 
                                               hypothesis_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Generate augmented point cloud from SBERT tokens using geometric operations
        """
        all_clouds = []
        
        # 1. Original premise tokens
        all_clouds.append(premise_tokens)
        
        # 2. Original hypothesis tokens
        all_clouds.append(hypothesis_tokens)
        
        # 3. Interpolated premise points
        premise_interp = self.generate_interpolated_points(premise_tokens, num_interpolations=40)
        if premise_interp.shape[0] > 0:
            all_clouds.append(premise_interp)
        
        # 4. Interpolated hypothesis points
        hyp_interp = self.generate_interpolated_points(hypothesis_tokens, num_interpolations=40)
        if hyp_interp.shape[0] > 0:
            all_clouds.append(hyp_interp)
        
        # 5. Jittered premise points
        premise_jitter = self.generate_jittered_points(premise_tokens, num_jitter=25)
        if premise_jitter.shape[0] > 0:
            all_clouds.append(premise_jitter)
        
        # 6. Jittered hypothesis points
        hyp_jitter = self.generate_jittered_points(hypothesis_tokens, num_jitter=25)
        if hyp_jitter.shape[0] > 0:
            all_clouds.append(hyp_jitter)
        
        # 7. Centroid variants for premise
        premise_centroid = self.generate_centroid_variants(premise_tokens, num_variants=15)
        if premise_centroid.shape[0] > 0:
            all_clouds.append(premise_centroid)
        
        # 8. Centroid variants for hypothesis
        hyp_centroid = self.generate_centroid_variants(hypothesis_tokens, num_variants=15)
        if hyp_centroid.shape[0] > 0:
            all_clouds.append(hyp_centroid)
        
        # 9. Cross-interpolation (premise-hypothesis pairs)
        combined_tokens = torch.cat([premise_tokens, hypothesis_tokens], dim=0)
        cross_interp = self.generate_interpolated_points(combined_tokens, num_interpolations=30)
        if cross_interp.shape[0] > 0:
            all_clouds.append(cross_interp)
        
        # Combine all clouds
        combined_cloud = torch.cat(all_clouds, dim=0)
        
        # Generate statistics
        stats = {
            'premise_points': premise_tokens.shape[0],
            'hypothesis_points': hypothesis_tokens.shape[0],
            'augmented_points': combined_cloud.shape[0] - premise_tokens.shape[0] - hypothesis_tokens.shape[0],
            'combined_total_points': combined_cloud.shape[0],
            'sufficient_for_phd': combined_cloud.shape[0] >= 100
        }
        
        return combined_cloud, stats


class AugmentedBaselineValidator:
    """Validator for augmented baseline clustering"""
    
    def __init__(self, 
                 val_data_path: str,
                 output_dir: str = "phd_method/augmented_baseline_results/",
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
        
        # Initialize augmented baseline generator
        self.point_cloud_generator = AugmentedBaselineGenerator()
        
        self.samples_per_class = 100
        self.min_points_for_phd = 100
        
        print(f"Augmented baseline validator initialized")
    
    def ph_dim_and_diagrams_from_distance_matrix(self, dm: np.ndarray,
                                 min_points: int = 50,
                                 max_points: int = 1000,
                                 point_jump: int = 25,
                                 h_dim: int = 0,
                                 alpha: float = 1.0) -> Tuple[float, List]:
        """Compute persistence on FULL point cloud"""
        assert dm.ndim == 2 and dm.shape[0] == dm.shape[1]
        
        # Compute persistence diagrams on FULL point cloud
        full_diagrams = ripser_parallel(dm, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        
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
        """Convert persistence diagrams to images"""
        
        all_birth_times = []
        all_lifespans = []
        
        for diagrams in all_diagrams:
            if diagrams is None:
                continue
            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    if diagram.ndim == 2 and diagram.shape[1] >= 2:
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        if len(finite_diagram) > 0:
                            all_birth_times.extend(finite_diagram[:, 0])
                            lifespans = finite_diagram[:, 1] - finite_diagram[:, 0]
                            all_lifespans.extend(lifespans)
        
        if len(all_lifespans) == 0:
            return []
        
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
        
        for diagrams in all_diagrams:
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
                                
                            except:
                                continue
            
            if has_content and combined_image.max() > 0:
                combined_image = combined_image / combined_image.max()
                persistence_images.append(combined_image.flatten())
        
        return persistence_images

    def compute_distance_matrix(self, point_cloud: torch.Tensor, metric: str = 'braycurtis') -> np.ndarray:
        """Compute distance matrix"""
        point_cloud_np = point_cloud.numpy()
        distance_matrix = pairwise_distances(point_cloud_np, metric=metric)
        return distance_matrix

    def filter_samples_by_token_count(self, samples: List[Dict], min_combined_tokens: int = 0) -> List[Dict]:
        """Filter samples"""
        filtered_samples = []
        
        for sample in samples:
            premise_tokens = sample['premise_tokens'].shape[0]
            hypothesis_tokens = sample['hypothesis_tokens'].shape[0]
            combined_tokens = premise_tokens + hypothesis_tokens
            
            if combined_tokens >= min_combined_tokens:
                filtered_samples.append(sample)
        
        return filtered_samples

    def generate_maximum_samples_by_class(self) -> Dict[str, List[Dict]]:
        """Generate maximum samples"""
        
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
            filtered_samples = self.filter_samples_by_token_count(class_samples)
            max_samples[class_name] = filtered_samples
        
        return max_samples

    def validate_augmented_baseline(self) -> ClusteringResult:
        """Main validation - augmented baseline"""
        
        print("\n" + "="*80)
        print("AUGMENTED BASELINE CLUSTERING (SBERT + GEOMETRIC AUGMENTATION)")
        print("="*80)
        
        max_samples = self.generate_maximum_samples_by_class()
            
        all_persistence_diagrams = []
        sample_labels = []
        ph_dim_values = {'entailment': [], 'neutral': [], 'contradiction': []}
        point_cloud_stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for class_idx, class_name in enumerate(['entailment', 'neutral', 'contradiction']):
            print(f"\nProcessing {class_name} samples...")
            
            class_samples = max_samples[class_name]

            for sample_idx, sample_data in enumerate(class_samples):
                premise_tokens = sample_data['premise_tokens']
                hypothesis_tokens = sample_data['hypothesis_tokens']
                
                # Generate augmented point cloud
                point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                    premise_tokens, hypothesis_tokens
                )
                
                point_cloud_stats[class_name].append(stats)
                
                if sample_idx % 500 == 0:
                    print(f"  Sample {sample_idx}: {stats['combined_total_points']} points " +
                          f"(original: {stats['premise_points'] + stats['hypothesis_points']}, " +
                          f"augmented: {stats['augmented_points']})")
                
                if not stats['sufficient_for_phd']:
                    continue
                
                distance_matrix = self.compute_distance_matrix(point_cloud)
                ph_dim, diagrams = self.ph_dim_and_diagrams_from_distance_matrix(distance_matrix)
                
                ph_dim_values[class_name].append(ph_dim)
                all_persistence_diagrams.append(diagrams)
                sample_labels.append(class_idx)

        print(f"\nCollected {len(all_persistence_diagrams)} diagram sets")

        persistence_images = self.persistence_diagrams_to_images(all_persistence_diagrams)
        
        if len(persistence_images) > 0:
            accuracy, sil_score, ari_score = self.perform_clustering_analysis(
                persistence_images, sample_labels
            )
        else:
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
                
                comprehensive_point_stats[class_name] = {
                    'combined_mean': float(np.mean(combined_points)),
                    'combined_std': float(np.std(combined_points)),
                    'sufficient_rate': float(np.mean([s['sufficient_for_phd'] for s in class_stats]))
                }
        
        result = ClusteringResult(
            model_name="SBERT_augmented_baseline",
            clustering_accuracy=accuracy,
            silhouette_score=sil_score,
            adjusted_rand_score=ari_score,
            num_samples=sum(len(samples) for samples in max_samples.values()),
            success=(accuracy > 0.7),
            ph_dim_values=ph_dim_values,
            ph_dim_stats=ph_dim_stats,
            point_cloud_stats=comprehensive_point_stats
        )
        
        print(f"\n" + "="*80)
        print("AUGMENTED BASELINE RESULTS")
        print("="*80)
        print(f"Clustering Accuracy: {accuracy:.3f}")
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        print(f"Success (>70%): {'🎉 YES' if result.success else '❌ NO'}")
        
        return result

    def perform_clustering_analysis(self, persistence_images: List[np.ndarray], 
                               sample_labels: List[int]) -> Tuple[float, float, float]:
        """Perform clustering"""
        
        if len(persistence_images) == 0:
            return 0.0, 0.0, 0.0
        
        X = np.array(persistence_images)
        y_true = np.array(sample_labels)
        
        n_clusters = len(np.unique(y_true))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
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
        
        return best_accuracy, silhouette, ari


def main():
    """Run augmented baseline validation"""
    
    val_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_test_sbert_tokens.pkl"
    output_dir = "MSc_Topology_Codebase/phd_method/augmented_baseline_results/"
    os.makedirs(output_dir, exist_ok=True)
    
    if not Path(val_data_path).exists():
        print(f"Validation data not found at: {val_data_path}")
        return
    
    validator = AugmentedBaselineValidator(
        val_data_path=val_data_path,
        output_dir=output_dir,
        seed=42
    )
    
    result = validator.validate_augmented_baseline()
    
    print("\n" + "="*80)
    print("COMPARISON:")
    print("  Pure SBERT baseline: ~42%")
    print(f"  SBERT + augmentation: {result.clustering_accuracy:.1%}")
    print("  SBERT + learned models: ~70%")
    print("="*80)


if __name__ == "__main__":
    main()