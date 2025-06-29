import torch
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import pdist, squareform, mahalanobis
from scipy.stats import multivariate_normal
import pickle
import time
from datetime import datetime
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phd_method.src_phd.topology import ph_dim_from_distance_matrix, fast_ripser, calculate_ph_dim

from src.order_embeddings_asymmetry import OrderEmbeddingModel

def flush_output():
    """Force output to appear immediately in SLURM"""
    sys.stdout.flush()
    sys.stderr.flush()

class SurfaceDistanceMetricAnalyzer:
    """
    Step 1.1: Surface Distance Metric Discovery using pre-processed embeddings
    
    Tests comprehensive set of distance metrics on different embedding spaces
    to find optimal metrics for entailment surface learning.
    """
    
    def __init__(self, 
                 bert_data_path: str,
                 order_model_path: str,
                 results_dir: str = 'entailment_surfaces/results/surface_analysis',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 seed: int = 42):
        """
        Initialize analyzer with pre-processed data paths
        
        Args:
            bert_data_path: Path to processed BERT embeddings (.pt file from text_processing.py)
            order_model_path: Path to trained order embedding model (.pt file from order_embeddings_asymmetry.py)
            results_dir: Directory to save analysis results
            device: Computing device
            seed: Random seed
        """
        self.bert_data_path = bert_data_path
        self.order_model_path = order_model_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.seed = seed
        
        # GPU OPTIMIZATION: Add efficiency settings
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.cuda.empty_cache()  # Clear memory
        
        # Comprehensive distance metrics for surface learning
        self.distance_metrics = [
            # Standard metrics
            'euclidean',        # L2 norm
            'manhattan',        # L1 norm  
            'chebyshev',        # L∞ norm
            'cosine',          # Cosine distance
            
            # Minkowski family
            'minkowski_3',     # L3 norm
            'minkowski_4',     # L4 norm
            
            # Other metrics
            'canberra',        # Weighted Manhattan
            'braycurtis',      # Normalized Manhattan
            
            # Custom metrics (implemented separately)
            'hyperbolic',      # Hyperbolic distance (if in hyperbolic space)
            'order_violation', # Order embedding violation energy
        ]

         # Embedding spaces for surface analysis (CORRECTED - only relational spaces)
        self.embedding_spaces = [
            'bert_concat',          # Concatenated [premise||hypothesis] - joint representation
            'bert_difference',      # Premise - Hypothesis (relationship vector)
            'order_concat',         # Concatenated order embeddings [order_premise||order_hypothesis]
            'order_difference',     # Order premise - hypothesis (order relationship)
            'order_violations',     # Order violation energies (inherently relational)
            'hyperbolic_concat',    # Concatenated hyperbolic embeddings
            'hyperbolic_distances', # Direct hyperbolic distances between P-H pairs (1D)
            'cone_features',        # Multiple cone-related features
        ]

        # PH-Dim parameters
        self.phd_params = {
            'min_points': 200,
            'max_points': 1000,
            'point_jump': 50,
            'h_dim': 0,
            'alpha': 1.0,
            'seed': seed
        }

        print(f"Surface Distance Metric Analyzer initialized")
        print(f"FOCUS: Testing distance metrics for premise-hypothesis pair relationships")
        print(f"Device: {self.device}")
        print(f"BERT data: {bert_data_path}")
        print(f"Order model: {order_model_path}")
        print(f"Distance metrics: {len(self.distance_metrics)} total")
        print(f"Relational embedding spaces: {len(self.embedding_spaces)} total")
        print(f"Each space represents premise-hypothesis pair relationships")
        flush_output()

        # Load data
        self._load_preprocessed_data()


    def _load_preprocessed_data(self):
        """Load pre-processed BERT embeddings and order model"""
        print("Loading pre-processed data...")

        # Load BERT embeddings
        if not os.path.exists(self.bert_data_path):
            raise FileNotFoundError(f"BERT data not found: {self.bert_data_path}")

        print(f"Loading BERT embeddings from {self.bert_data_path}")
        self.bert_data = torch.load(self.bert_data_path, map_location=self.device, weights_only=False)

        print(f"BERT data loaded:")
        print(f"  Premise embeddings: {self.bert_data['premise_embeddings'].shape}")
        print(f"  Hypothesis embeddings: {self.bert_data['hypothesis_embeddings'].shape}")
        print(f"  Labels: {len(self.bert_data['labels'])}")
        print(f"  Label distribution: {self.bert_data['metadata']['label_counts']}")
        
        # Load order embedding model
        if not os.path.exists(self.order_model_path):
            raise FileNotFoundError(f"Order model not found: {self.order_model_path}")
        
        print(f"Loading order model from {self.order_model_path}")
        checkpoint = torch.load(self.order_model_path, map_location=self.device, weights_only=False)

        model_config = checkpoint['model_config']
        self.order_model = OrderEmbeddingModel(
            bert_dim=model_config['bert_dim'],
            order_dim=model_config['order_dim'],
            asymmetry_weight=model_config.get('asymmetry_weight', 0.2)
        )
        self.order_model.load_state_dict(checkpoint['model_state_dict'])
        self.order_model.to(self.device)
        self.order_model.eval()
        
        print(f"Order model loaded (validation loss: {checkpoint.get('best_val_loss', 'N/A')})")

        # Initialize hyperbolic and cone pipeline
        try:
            from src.hyperbolic_projection_asymmetry import HyperbolicOrderEmbeddingPipeline
            self.hyperbolic_pipeline = HyperbolicOrderEmbeddingPipeline(self.order_model_path)
            print("Hyperbolic pipeline loaded successfully")
        except Exception as e:
            print(f"Could not load hyperbolic pipeline: {e}")
            raise
       
        try:
            from src.entailment_cones_asymmetry import EnhancedHyperbolicConeEmbeddingPipeline
            self.cone_pipeline = EnhancedHyperbolicConeEmbeddingPipeline(self.order_model_path)
            print("Enhanced cone pipeline loaded successfully")
        except Exception as e:
            print(f"Could not load cone pipeline: {e}")
            raise        


    def extract_all_embedding_spaces(self, max_samples_per_class: int = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract all embedding spaces from pre-processed data
        
        Args:
            max_samples_per_class: Limit samples per class (None for all)
            
        Returns:
            Dict mapping space names to class embeddings
        """
        print("Extracting all embedding spaces...")

        # Organize data by class - OPTIMIZATION: Keep on GPU
        premise_embs = self.bert_data['premise_embeddings']  # Already on GPU
        hypothesis_embs = self.bert_data['hypothesis_embeddings']  # Already on GPU
        labels = self.bert_data['labels']

        # Group by class - OPTIMIZATION: Use tensor operations
        data_by_class = {'entailment': {}, 'neutral': {}, 'contradiction': {}}

        for label in data_by_class.keys():
            # OPTIMIZATION: Create mask on GPU
            mask = torch.tensor([l == label for l in labels], device=self.device, dtype=torch.bool)
            indices = torch.where(mask)[0]
            
            if max_samples_per_class and len(indices) > max_samples_per_class:
                # OPTIMIZATION: Random sampling on GPU
                perm = torch.randperm(len(indices), device=self.device)[:max_samples_per_class]
                indices = indices[perm]
            
            # OPTIMIZATION: Keep data on GPU
            data_by_class[label] = {
                'premise_bert': premise_embs[indices],
                'hypothesis_bert': hypothesis_embs[indices],
                'indices': indices
            }
            
            print(f"  {label}: {len(indices)} samples")

        # STEP 1: Generate EUCLIDEAN order embeddings (before hyperbolic projection)
        print("Computing EUCLIDEAN order embeddings...")
        with torch.no_grad():
            for label in data_by_class.keys():
                premise_bert = data_by_class[label]['premise_bert']
                hypothesis_bert = data_by_class[label]['hypothesis_bert']
                
                # OPTIMIZATION: Batch compute on GPU
                premise_order_euclidean = self.order_model(premise_bert)
                hypothesis_order_euclidean = self.order_model(hypothesis_bert)
                
                # OPTIMIZATION: Vectorized violation computation on GPU
                euclidean_order_violations = self.order_model.order_violation_energy(
                    premise_order_euclidean, hypothesis_order_euclidean
                )
                
                data_by_class[label].update({
                    'premise_order_euclidean': premise_order_euclidean,
                    'hypothesis_order_euclidean': hypothesis_order_euclidean,
                    'euclidean_order_violations': euclidean_order_violations
                })

        # STEP 2: Generate HYPERBOLIC features (enhanced cone pipeline as single source of truth)
        if self.cone_pipeline:
            print("Computing ALL HYPERBOLIC features from enhanced cone pipeline...")
            with torch.no_grad():
                for label in data_by_class.keys():
                    premise_bert = data_by_class[label]['premise_bert']
                    hypothesis_bert = data_by_class[label]['hypothesis_bert']
                    
                    try:
                        # OPTIMIZATION: Process in batches for large datasets
                        batch_size = min(1000, len(premise_bert))
                        
                        if len(premise_bert) <= batch_size:
                            # Small enough to process at once
                            enhanced_results = self.cone_pipeline.compute_enhanced_cone_energies(
                                premise_bert, hypothesis_bert
                            )
                        else:
                            # Process in batches
                            all_results = {}
                            for i in range(0, len(premise_bert), batch_size):
                                batch_premise = premise_bert[i:i+batch_size]
                                batch_hypothesis = hypothesis_bert[i:i+batch_size]
                                
                                batch_results = self.cone_pipeline.compute_enhanced_cone_energies(
                                    batch_premise, batch_hypothesis
                                )
                                
                                # Accumulate results
                                for key, value in batch_results.items():
                                    if key not in all_results:
                                        all_results[key] = []
                                    all_results[key].append(value)
                            
                            # OPTIMIZATION: Concatenate on GPU
                            enhanced_results = {}
                            for key, value_list in all_results.items():
                                enhanced_results[key] = torch.cat(value_list, dim=0)

                        
                        # Store ALL hyperbolic features from enhanced cone pipeline
                        data_by_class[label].update({
                            # HYPERBOLIC order embeddings (after hyperbolic projection)
                            'premise_order_hyperbolic': enhanced_results['premise_hyperbolic'],
                            'hypothesis_order_hyperbolic': enhanced_results['hypothesis_hyperbolic'],
                            
                            # HYPERBOLIC order energies (computed in hyperbolic space)
                            'hyperbolic_order_energies': enhanced_results['order_energies'],
                            'forward_order_energies': enhanced_results.get('forward_energies', enhanced_results['order_energies']),
                            'backward_order_energies': enhanced_results.get('backward_energies', torch.zeros_like(enhanced_results['order_energies'])),
                            'asymmetric_order_energies': enhanced_results.get('asymmetric_energies', torch.zeros_like(enhanced_results['order_energies'])),
                            
                            # Hyperbolic geometric features
                            'hyperbolic_distances': enhanced_results['hyperbolic_distances'],
                            
                            # Cone energies (computed in hyperbolic space)
                            'cone_energies': enhanced_results['cone_energies'],
                            'forward_cone_energies': enhanced_results.get('forward_cone_energies', enhanced_results['cone_energies']),
                            'backward_cone_energies': enhanced_results.get('backward_cone_energies', torch.zeros_like(enhanced_results['cone_energies'])),
                            'cone_asymmetries': enhanced_results.get('cone_asymmetries', torch.zeros_like(enhanced_results['cone_energies'])),
                            
                            # Combined enhanced features (hyperbolic)
                            'enhanced_cone_features': torch.cat([
                                enhanced_results['cone_energies'].unsqueeze(1).to(self.device),
                                enhanced_results.get('forward_cone_energies', enhanced_results['cone_energies']).unsqueeze(1).to(self.device),
                                enhanced_results.get('backward_cone_energies', torch.zeros_like(enhanced_results['cone_energies'])).unsqueeze(1).to(self.device),
                                enhanced_results['order_energies'].unsqueeze(1).to(self.device),
                                enhanced_results.get('asymmetric_energies', torch.zeros_like(enhanced_results['order_energies'])).unsqueeze(1).to(self.device)
                            ], dim=1)
                        })
                        
                    except Exception as e:
                        print(f"Error computing enhanced cone features for {label}: {e}")
                        raise
        else:
            print("Enhanced cone pipeline not available - hyperbolic features will be skipped")

        
        # Extract all embedding spaces (CORRECTED - only relational embeddings)
        all_embeddings = {}

        for space in self.embedding_spaces:
            print(f"Extracting {space}...")
            space_embeddings = {}
            
            for label in data_by_class.keys():
                if space == 'bert_concat':
                    # OPTIMIZATION: GPU concatenation
                    space_embeddings[label] = torch.cat([
                        data_by_class[label]['premise_bert'],
                        data_by_class[label]['hypothesis_bert']
                    ], dim=1)
                    
                elif space == 'bert_difference':
                    # OPTIMIZATION: GPU subtraction
                    space_embeddings[label] = data_by_class[label]['premise_bert'] - data_by_class[label]['hypothesis_bert']
                    
                elif space == 'order_concat':
                    # Concatenated EUCLIDEAN order embeddings
                    if 'premise_order_euclidean' in data_by_class[label]:
                        space_embeddings[label] = torch.cat([
                            data_by_class[label]['premise_order_euclidean'],
                            data_by_class[label]['hypothesis_order_euclidean']
                        ], dim=1)
                    else:
                        print(f"    Euclidean order embeddings not available for {space}")
                        continue
                        
                elif space == 'order_difference':
                    # EUCLIDEAN order relationship vector
                    if 'premise_order_euclidean' in data_by_class[label]:
                        space_embeddings[label] = data_by_class[label]['premise_order_euclidean'] - data_by_class[label]['hypothesis_order_euclidean']
                    else:
                        print(f"    Euclidean order embeddings not available for {space}")
                        continue
                        
                elif space == 'order_violations':
                    # EUCLIDEAN order violation energies
                    if 'euclidean_order_violations' in data_by_class[label]:
                        space_embeddings[label] = data_by_class[label]['euclidean_order_violations'].unsqueeze(1)
                    else:
                        print(f"    Euclidean order violations not available for {space}")
                        continue
                        
                elif space == 'hyperbolic_concat':
                    # Concatenated HYPERBOLIC order embeddings
                    if 'premise_order_hyperbolic' in data_by_class[label] and data_by_class[label]['premise_order_hyperbolic'] is not None:
                        space_embeddings[label] = torch.cat([
                            data_by_class[label]['premise_order_hyperbolic'],
                            data_by_class[label]['hypothesis_order_hyperbolic']
                        ], dim=1)
                    else:
                        print(f"    Hyperbolic order embeddings not available for {space}")
                        continue
                        
                elif space == 'hyperbolic_distances':
                    if 'hyperbolic_distances' in data_by_class[label] and data_by_class[label]['hyperbolic_distances'] is not None:
                        # Direct hyperbolic distances between premise-hypothesis pairs
                        space_embeddings[label] = data_by_class[label]['hyperbolic_distances'].unsqueeze(1)
                    else:
                        print(f"    Hyperbolic distances not available for {space}")
                        continue
                        
                elif space == 'cone_features':
                    if 'enhanced_cone_features' in data_by_class[label] and data_by_class[label]['enhanced_cone_features'] is not None:
                        # Combined enhanced features (multi-dimensional)
                        space_embeddings[label] = data_by_class[label]['enhanced_cone_features']
                    else:
                        print(f"    Enhanced cone features not available for {space}")
                        continue
            
            # Only add space if we have embeddings
            if space_embeddings:
                all_embeddings[space] = space_embeddings

                # Print shapes for available embeddings
                for label, embs in space_embeddings.items():
                    print(f"  {space} {label}: {embs.shape}")
                    flush_output()

        return all_embeddings


    def compute_distance_matrix_advanced(self, embeddings: torch.Tensor, metric: str) -> np.ndarray:
        """
        Compute distance matrix using advanced metrics
        
        Args:
            embeddings: Embedding tensor [n_samples, embed_dim]
            metric: Distance metric name
            
        Returns:
            Distance matrix [n_samples, n_samples]
        """
        # OPTIMIZATION: Only convert to numpy when necessary
        embeddings_np = embeddings.detach().cpu().numpy()
        n_samples = embeddings_np.shape[0]
        
        # Standard sklearn metrics
        sklearn_metrics = [
            'euclidean', 'manhattan', 'chebyshev', 'cosine', 
            'correlation', 'braycurtis', 'canberra']
        
        if metric in sklearn_metrics:
            return pairwise_distances(embeddings_np, metric=metric)
        
        # Minkowski metrics
        elif metric == 'minkowski_3':
            return pairwise_distances(embeddings_np, metric='minkowski', p=3)
        elif metric == 'minkowski_4':
            return pairwise_distances(embeddings_np, metric='minkowski', p=4)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_topology_analysis(self, embeddings: torch.Tensor, metric: str, 
                                 class_name: str, space_name: str) -> float:
        """
        Compute PH-Dim for surface topology analysis using appropriate topology function
        
        Args:
            embeddings: Class embeddings
            metric: Distance metric
            class_name: Entailment class name
            space_name: Embedding space name
            
        Returns:
            PH-Dim value
        """
        print(f"  Computing PH-Dim for {class_name} in {space_name} using {metric}")
        flush_output()
        
        if len(embeddings) < self.phd_params['min_points']:
            print(f"    Warning: Only {len(embeddings)} samples, need ≥{self.phd_params['min_points']}")
            return np.nan
        
        try:
            # OPTIMIZATION: Subsample on GPU first, then convert to numpy
            max_points = min(self.phd_params['max_points'], len(embeddings))
            if len(embeddings) > max_points:
                indices = torch.randperm(len(embeddings), device=self.device)[:max_points]
                embeddings = embeddings[indices]
            
            embeddings_np = embeddings.detach().cpu().numpy()
            
            # Sklearn-supported metrics: use fast_ripser for efficiency
            sklearn_metrics = [
                'euclidean', 'manhattan', 'chebyshev', 'cosine', 
                'correlation', 'braycurtis', 'canberra', 'hamming'
            ]
            
            # Minkowski metrics
            minkowski_metrics = ['minkowski_3', 'minkowski_4']
            
            if metric in sklearn_metrics:
                # Use fast_ripser which supports sklearn metrics directly
                phd = fast_ripser(
                    embeddings_np,
                    min_points=self.phd_params['min_points'],
                    max_points=min(self.phd_params['max_points'], len(embeddings)),
                    point_jump=self.phd_params['point_jump'],
                    h_dim=self.phd_params['h_dim'],
                    alpha=self.phd_params['alpha'],
                    seed=self.phd_params['seed'],
                    metric=metric
                )
                
            elif metric in minkowski_metrics:
                # Handle Minkowski metrics with custom p values
                distance_matrix = self.compute_distance_matrix_advanced(embeddings, metric)
                
                phd = ph_dim_from_distance_matrix(
                    distance_matrix,
                    min_points=self.phd_params['min_points'],
                    max_points=min(self.phd_params['max_points'], len(embeddings)),
                    point_jump=self.phd_params['point_jump'],
                    h_dim=self.phd_params['h_dim'],
                    alpha=self.phd_params['alpha'],
                    seed=self.phd_params['seed']
                )
                
            elif metric in ['hyperbolic', 'order_violation']:
                phd = calculate_ph_dim(
                    embeddings_np,
                    min_points=self.phd_params['min_points'],
                    max_points=min(self.phd_params['max_points'], len(embeddings)),
                    point_jump=self.phd_params['point_jump'],
                    h_dim=self.phd_params['h_dim'],
                    metric=None,  # Let ripser handle raw points
                    alpha=self.phd_params['alpha'],
                    seed=self.phd_params['seed']
                    # Custom distance will be computed internally by ripser as needed
                )
                
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Check for valid result
            if np.any(np.isnan([phd])) or np.any(np.isinf([phd])):
                print(f"    Invalid PH-Dim result for {metric}, skipping")
                return np.nan
            
            print(f"    {class_name} PH-Dim: {phd:.4f}")
            return phd
            
        except Exception as e:
            print(f"    Error computing PH-Dim for {class_name}: {e}")
            return np.nan


    def compute_cross_class_surface_analysis(self, embeddings_by_class: Dict[str, torch.Tensor], 
                                           metric: str, space_name: str) -> Dict[str, float]:
        """
        STEP 2: Analyze cross-class distances for surface learning
        
        This measures distances BETWEEN different entailment classes to find
        metrics suitable for learning entailment surfaces.
        
        Args:
            embeddings_by_class: Embeddings organized by entailment class
            metric: Distance metric to test
            space_name: Embedding space name
            
        Returns:
            Dictionary with cross-class distance metrics
        """
        print(f"  Computing cross-class surface analysis for {space_name} using {metric}")
        flush_output()
        
        # Check if we have all three classes
        required_classes = {'entailment', 'neutral', 'contradiction'}
        available_classes = set(embeddings_by_class.keys())
        
        if not required_classes.issubset(available_classes):
            print(f"    Missing classes for cross-class analysis: {required_classes - available_classes}")
            return {}
        
        try:
            # OPTIMIZATION: Compute centroids on GPU first
            centroids_gpu = {}
            for label, embeddings in embeddings_by_class.items():
                centroids_gpu[label] = torch.mean(embeddings, dim=0)  # Keep on GPU
            
            # 1. CENTROID DISTANCES - How far apart are class centers?
            centroid_distances = {}
            for label1 in required_classes:
                for label2 in required_classes:
                    if label1 != label2:
                        c1, c2 = centroids_gpu[label1], centroids_gpu[label2]
                        
                        # OPTIMIZATION: Compute distance on GPU when possible
                        if metric == 'euclidean':
                            dist = torch.norm(c1 - c2).item()
                        elif metric == 'cosine':
                            dist = (1 - torch.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0))).item()
                        elif metric == 'manhattan':
                            dist = torch.sum(torch.abs(c1 - c2)).item()
                        elif metric in ['chebyshev', 'canberra', 'braycurtis']:
                            # Convert to numpy only when necessary
                            c1_np, c2_np = c1.cpu().numpy(), c2.cpu().numpy()
                            dist = pairwise_distances([c1_np], [c2_np], metric=metric)[0, 0]
                        elif metric.startswith('minkowski'):
                            p_value = int(metric.split('_')[1])
                            c1_np, c2_np = c1.cpu().numpy(), c2.cpu().numpy()
                            dist = pairwise_distances([c1_np], [c2_np], metric='minkowski', p=p_value)[0, 0]
                        elif metric == 'hyperbolic':
                            # Use hyperbolic distance for centroids
                            c1_np, c2_np = c1.cpu().numpy(), c2.cpu().numpy()
                            dist = self._hyperbolic_distance_pair(c1_np, c2_np)
                        elif metric == 'order_violation':
                            # Use order violation distance for centroids
                            c1_np, c2_np = c1.cpu().numpy(), c2.cpu().numpy()
                            dist = self._order_violation_distance_pair(c1_np, c2_np)
                        else:
                            # Fallback to euclidean on GPU
                            dist = torch.norm(c1 - c2).item()
                        
                        centroid_distances[f'{label1}_to_{label2}'] = dist
            
            # 2. MINIMUM CROSS-CLASS DISTANCES - Closest points between classes
            min_distances = {}
            max_distances = {}
            
            for label1 in required_classes:
                for label2 in required_classes:
                    if label1 != label2:
                        # OPTIMIZATION: Sample subset for efficiency (cross-class distance computation is expensive)
                        sample_size = min(500, len(embeddings_by_class[label1]), len(embeddings_by_class[label2]))
                        
                        if sample_size < 10:
                            continue
                        
                        # OPTIMIZATION: Random sampling on GPU
                        embs1 = embeddings_by_class[label1]
                        embs2 = embeddings_by_class[label2]
                        
                        idx1 = torch.randperm(len(embs1), device=self.device)[:sample_size]
                        idx2 = torch.randperm(len(embs2), device=self.device)[:sample_size]
                        
                        embs1_sample = embs1[idx1]
                        embs2_sample = embs2[idx2]
                        
                        # Convert to numpy only when necessary
                        embs1_sample_np = embs1_sample.cpu().numpy()
                        embs2_sample_np = embs2_sample.cpu().numpy()
                        
                        # Compute cross-class distance matrix
                        if metric in ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'correlation', 'braycurtis', 'canberra', 'hamming']:
                            cross_distances = pairwise_distances(embs1_sample_np, embs2_sample_np, metric=metric)
                        elif metric.startswith('minkowski'):
                            p_value = int(metric.split('_')[1])
                            cross_distances = pairwise_distances(embs1_sample_np, embs2_sample_np, metric='minkowski', p=p_value)
                        elif metric == 'hyperbolic':
                            # Use hyperbolic distance implementation
                            cross_distances = self._compute_hyperbolic_cross_distances(embs1_sample_np, embs2_sample_np)
                        elif metric == 'order_violation':
                            # Use order violation distance implementation
                            cross_distances = self._compute_order_violation_cross_distances(embs1_sample_np, embs2_sample_np)
                        else:
                            # Fallback for unknown metrics
                            cross_distances = pairwise_distances(embs1_sample_np, embs2_sample_np, metric='euclidean')
                        
                        min_distances[f'{label1}_to_{label2}'] = np.min(cross_distances)
                        max_distances[f'{label1}_to_{label2}'] = np.max(cross_distances)

            # 3. SURFACE SEPARATION QUALITY - Key metrics for surface learning
            
            # Entailment separation score - How well separated is entailment from others?
            entailment_separation = 0.0
            if 'entailment_to_neutral' in centroid_distances and 'entailment_to_contradiction' in centroid_distances:
                # Average distance from entailment to other classes
                avg_entailment_distance = (centroid_distances['entailment_to_neutral'] + 
                                         centroid_distances['entailment_to_contradiction']) / 2
                
                # Compare to within-class spread (approximate) - OPTIMIZATION: Compute on GPU
                entailment_embs = embeddings_by_class['entailment']
                entailment_centroid = centroids_gpu['entailment']
                entailment_spread = torch.std(torch.norm(entailment_embs - entailment_centroid, dim=1)).item()
                
                if entailment_spread > 0:
                    entailment_separation = avg_entailment_distance / entailment_spread

            # Surface gradient score - Is there a clear ordering: entailment < neutral < contradiction?
            surface_gradient = 0.0
            if ('entailment_to_neutral' in centroid_distances and 
                'entailment_to_contradiction' in centroid_distances and
                'neutral_to_contradiction' in centroid_distances):
                
                ent_to_neutral = centroid_distances['entailment_to_neutral']
                ent_to_contradiction = centroid_distances['entailment_to_contradiction']
                neutral_to_contradiction = centroid_distances['neutral_to_contradiction']
                
                # Ideal ordering: ent_to_neutral < ent_to_contradiction
                if ent_to_neutral < ent_to_contradiction:
                    surface_gradient += 1.0
                
                # Bonus if neutral is truly between entailment and contradiction
                if ent_to_neutral < neutral_to_contradiction < ent_to_contradiction:
                    surface_gradient += 0.5
            
            # 4. OVERALL SURFACE LEARNING SCORE
            surface_learning_score = (entailment_separation / 10.0 + surface_gradient) / 2.0
            
            results = {
                # Centroid distances
                'centroid_ent_to_neutral': centroid_distances.get('entailment_to_neutral', 0),
                'centroid_ent_to_contradiction': centroid_distances.get('entailment_to_contradiction', 0),
                'centroid_neutral_to_contradiction': centroid_distances.get('neutral_to_contradiction', 0),
                
                # Minimum cross-class distances
                'min_ent_to_neutral': min_distances.get('entailment_to_neutral', 0),
                'min_ent_to_contradiction': min_distances.get('entailment_to_contradiction', 0),
                
                # Surface learning metrics
                'entailment_separation': entailment_separation,
                'surface_gradient': surface_gradient,
                'surface_learning_score': surface_learning_score
            }
            
            print(f"    Entailment separation: {entailment_separation:.4f}")
            print(f"    Surface gradient: {surface_gradient:.4f}")
            print(f"    Surface learning score: {surface_learning_score:.4f}")
            
            return results
            
        except Exception as e:
            print(f"    Error in cross-class analysis: {e}")
            return {}

    def _hyperbolic_distance_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        """Efficient hyperbolic distance for single pair"""
        # Ensure points are in unit ball
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        
        if x_norm >= 0.99:
            x = x * (0.99 / x_norm)
        if y_norm >= 0.99:
            y = y * (0.99 / y_norm)
        
        # Hyperbolic distance formula
        diff = x - y
        euclidean_dist_sq = np.dot(diff, diff)
        
        numerator = 2 * euclidean_dist_sq
        denominator = (1 - np.dot(x, x)) * (1 - np.dot(y, y))
        
        if denominator > 1e-10:
            ratio = 1 + numerator / denominator
            if ratio >= 1:
                return np.arccosh(ratio)
        
        return 0.0  # Fallback

    def _order_violation_distance_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        """Efficient order violation distance for single pair"""
        if len(x.shape) == 0 or x.shape[0] == 1:  # 1D case
            return abs(float(x) - float(y))
        else:
            # Multi-dimensional case
            violation_x_to_y = np.maximum(0, x - y)
            violation_y_to_x = np.maximum(0, y - x)
            return np.sum(violation_x_to_y) + np.sum(violation_y_to_x)

    def _compute_hyperbolic_cross_distances(self, embs1: np.ndarray, embs2: np.ndarray) -> np.ndarray:
        """Compute hyperbolic cross-distances between two sets of embeddings"""
        n1, n2 = len(embs1), len(embs2)
        cross_distances = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                cross_distances[i, j] = self._hyperbolic_distance_pair(embs1[i], embs2[j])
        
        return cross_distances

    def _compute_order_violation_cross_distances(self, embs1: np.ndarray, embs2: np.ndarray) -> np.ndarray:
        """Compute order violation cross-distances between two sets of embeddings"""
        n1, n2 = len(embs1), len(embs2)
        cross_distances = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                cross_distances[i, j] = self._order_violation_distance_pair(embs1[i], embs2[j])
        
        return cross_distances


    def evaluate_surface_separation_quality(self, phd_scores: Dict[str, float]) -> Dict[str, float]:
        """Evaluate how well this metric separates entailment classes - CORRECTED: Focus on maximum class separation"""
        valid_scores = {k: v for k, v in phd_scores.items() if not np.isnan(v)}
        
        if len(valid_scores) < 2:
            return {
                'surface_quality': 0.0, 
                'separation_ratio': 0.0, 
                'entailment_simplicity': 0.0,
                'class_distinctiveness': 0.0,
                'overall_separation': 0.0
            }
        
        scores = list(valid_scores.values())
        
        # Basic separation metrics
        separation_ratio = max(scores) / min(scores) if min(scores) > 0 else 0.0
        min_max_diff = max(scores) - min(scores)
        std_dev = np.std(scores)
        
        # Entailment simplicity - test if entailment has simpler topology than others
        entailment_simplicity = 0.0
        if 'entailment' in valid_scores:
            entailment_phd = valid_scores['entailment']
            other_phds = [v for k, v in valid_scores.items() if k != 'entailment']
            if other_phds:
                entailment_simplicity = sum(1 for phd in other_phds if entailment_phd < phd) / len(other_phds)
        
        # Class distinctiveness - how well separated are ALL classes from each other
        class_distinctiveness = 0.0
        if len(valid_scores) >= 3:
            class_list = list(valid_scores.keys())
            pairwise_diffs = []
            
            for i in range(len(class_list)):
                for j in range(i+1, len(class_list)):
                    diff = abs(valid_scores[class_list[i]] - valid_scores[class_list[j]])
                    pairwise_diffs.append(diff)
            
            if pairwise_diffs:
                avg_pairwise_diff = np.mean(pairwise_diffs)
                max_possible_diff = max(scores) - min(scores)
                if max_possible_diff > 0:
                    class_distinctiveness = avg_pairwise_diff / max_possible_diff
        elif len(valid_scores) == 2:
            max_possible_diff = max(scores) - min(scores)
            class_distinctiveness = 1.0 if max_possible_diff > 0 else 0.0
        
        # Overall separation quality combines multiple factors
        overall_separation = (
            separation_ratio / 20.0 +      # Ratio component (normalized)
            min_max_diff / 10.0 +          # Absolute difference component  
            std_dev / 5.0 +                # Standard deviation component
            class_distinctiveness          # Pairwise distinctiveness
        ) / 4.0
        
        # Surface quality for backward compatibility
        surface_quality = (
            separation_ratio/20.0 + 
            min_max_diff/10.0 + 
            entailment_simplicity + 
            class_distinctiveness
        ) / 4.0
        
        return {
            'surface_quality': surface_quality,
            'separation_ratio': separation_ratio,
            'entailment_simplicity': entailment_simplicity,
            'class_distinctiveness': class_distinctiveness,
            'overall_separation': overall_separation,
            'min_max_diff': min_max_diff,
            'std_dev': std_dev
        }

    

    def run_comprehensive_analysis(self):
        """Run comprehensive surface distance metric analysis with GPU optimizations"""
        print("="*80)
        print("COMPREHENSIVE SURFACE DISTANCE METRIC ANALYSIS (GPU-OPTIMIZED)")
        print("Testing all distance metrics across all embedding spaces")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated")
        print("="*80)
        
        # Extract all embedding spaces
        all_embeddings = self.extract_all_embedding_spaces()
        
        # Results storage
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test each embedding space
        for space_name, space_embeddings in all_embeddings.items():
            print(f"\n{'='*60}")
            print(f"TESTING SPACE: {space_name}")
            print(f"{'='*60}")
            
            # OPTIMIZATION: Memory cleanup between spaces
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            space_results = {}
            
            # Test each distance metric
            for metric in self.distance_metrics:
                print(f"\n--- Testing {metric} metric ---")
                flush_output()
                
                metric_results = {}
                phd_scores = {}
                
                # STEP 1: Compute PH-Dim for each class (within-class topology)
                for class_name, embeddings in space_embeddings.items():
                    phd = self.compute_topology_analysis(
                        embeddings, metric, class_name, space_name
                    )
                    phd_scores[class_name] = phd
                    metric_results[f'phd_{class_name}'] = phd
                
                # Evaluate surface separation quality (from Step 1)
                surface_metrics = self.evaluate_surface_separation_quality(phd_scores)
                metric_results.update(surface_metrics)
                
                # STEP 2: Compute cross-class distances for surface learning
                cross_class_metrics = self.compute_cross_class_surface_analysis(
                    space_embeddings, metric, space_name
                )
                metric_results.update(cross_class_metrics)
                
                space_results[metric] = metric_results
                
                print(f"  STEP 1 - PH-Dim scores: {phd_scores}")
                print(f"  STEP 1 - Surface quality: {surface_metrics['surface_quality']:.4f}")
                print(f"  STEP 2 - Cross-class surface score: {cross_class_metrics.get('surface_learning_score', 0):.4f}")
                
                # Combined score for ranking
                combined_score = (surface_metrics['surface_quality'] + 
                                cross_class_metrics.get('surface_learning_score', 0)) / 2.0
                metric_results['combined_surface_score'] = combined_score
                print(f"  COMBINED surface learning potential: {combined_score:.4f}")
            
            all_results[space_name] = space_results
            
            # Save intermediate results
            results_file = self.results_dir / f"surface_analysis_{space_name}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(space_results, f, indent=2, default=str)
            print(f"\nSaved {space_name} results to {results_file}")
        
        # Save complete results and generate comprehensive report
        final_results_file = self.results_dir / f"comprehensive_surface_analysis_{timestamp}.json"
        with open(final_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self._generate_simple_report(all_results, timestamp)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        if torch.cuda.is_available():
            print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated")
        print(f"Results saved to: {final_results_file}")
        print(f"{'='*80}")
        
        return all_results


    def _generate_simple_report(self, results: Dict, timestamp: str):
        """Generate simple plain output report for analysis"""
        report_file = self.results_dir / f"simple_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("SURFACE DISTANCE METRIC ANALYSIS - PLAIN RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for space_name, space_results in results.items():
                f.write(f"EMBEDDING SPACE: {space_name}\n")
                f.write("-" * 50 + "\n")
                
                for metric, metric_results in space_results.items():
                    f.write(f"\nMetric: {metric}\n")
                    
                    # PH-Dim results
                    f.write("  PH-Dim Analysis:\n")
                    for key, value in metric_results.items():
                        if key.startswith('phd_'):
                            f.write(f"    {key}: {value}\n")
                    
                    # Surface separation quality
                    f.write("  Surface Separation Quality:\n")
                    separation_keys = ['surface_quality', 'separation_ratio', 'entailment_simplicity', 
                                     'class_distinctiveness', 'overall_separation', 'min_max_diff', 'std_dev']
                    for key in separation_keys:
                        if key in metric_results:
                            f.write(f"    {key}: {metric_results[key]}\n")
                    
                    # Cross-class analysis
                    f.write("  Cross-Class Surface Analysis:\n")
                    cross_class_keys = ['centroid_ent_to_neutral', 'centroid_ent_to_contradiction', 
                                      'centroid_neutral_to_contradiction', 'min_ent_to_neutral', 
                                      'min_ent_to_contradiction', 'entailment_separation', 
                                      'class_separation_quality', 'surface_learning_score']
                    for key in cross_class_keys:
                        if key in metric_results:
                            f.write(f"    {key}: {metric_results[key]}\n")
                    
                    # Combined score
                    if 'combined_surface_score' in metric_results:
                        f.write(f"  Combined Surface Score: {metric_results['combined_surface_score']}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Simple report saved to: {report_file}")
    

def main():
    """Main execution function"""

    # Initialize analyzer
    analyzer = SurfaceDistanceMetricAnalyzer(
        bert_data_path="data/processed/snli_full_standard_BERT.pt",
        order_model_path="models/enhanced_order_embeddings_snli_full.pt",
        results_dir="entailment_surfaces/results",
        seed=42
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    return results


if __name__ == "__main__":
    main()
                            