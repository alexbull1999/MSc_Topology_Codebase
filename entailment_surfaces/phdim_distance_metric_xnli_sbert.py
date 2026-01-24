# entailment_surfaces/phdim_distance_metric_xnli_single.py
"""
PH-Dimension Analysis for XNLI Multilingual Data - Single Language, Single Seed
Simplified version testing only Bray-Curtis and Cosine on sbert_concat and lattice_containment
Run separately for each language and seed combination
Now computes BOTH H0 and H1 PH-Dimensions using the robust clustering method
"""

import torch
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
from gph.python import ripser_parallel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def flush_output():
    """Force output to appear immediately in SLURM"""
    sys.stdout.flush()
    sys.stderr.flush()

def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                            min_points=200,
                                            max_points=1000,
                                            point_jump=50,
                                            h_dim=0,
                                            alpha: float = 1.,
                                            seed: int = 42) -> Tuple[float, List]:
    """
    Compute PH-dimension and persistence diagrams from distance matrix
    This is the robust version used in the 100% accuracy clustering code
    """
    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape

    # Set seed for reproducible sampling
    np.random.seed(seed)

    test_n = range(min_points, max_points, point_jump)
    lengths = []
    all_diagrams = []

    for points_number in test_n:
        if points_number >= dm.shape[0]:
            break
            
        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]

        diagrams = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        all_diagrams.append(diagrams)

        # Handle case where there might be no features in this dimension
        if h_dim < len(diagrams):
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            if len(d) > 0:
                lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())
            else:
                lengths.append(0.0)  # No features = 0 lifetime sum
        else:
            lengths.append(0.0)

    if len(lengths) < 2:
        return np.nan, all_diagrams

    lengths = np.array(lengths)
    
    # Avoid log(0) issues
    if np.any(lengths <= 0):
        print(f"    Warning: Some length values are zero or negative for h_dim={h_dim}")
        # Replace zeros with small epsilon for log calculation
        lengths = np.maximum(lengths, 1e-10)

    x = np.log(np.array(list(test_n[:len(lengths)])))
    y = np.log(lengths)
    N = len(x)
    
    if N < 2:
        return np.nan, all_diagrams
    
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    
    if m == 1:
        return np.nan, all_diagrams
    
    ph_dimension = alpha / (1 - m)
    
    return ph_dimension, all_diagrams


class XNLIPHDimAnalyzer:
    """
    PH-Dimension analysis for XNLI multilingual data
    Tests only Bray-Curtis and Cosine distance on sbert_concat and lattice_containment
    Single language, single seed per run
    Computes both H0 and H1 persistent homology dimensions using robust method
    """
    
    def __init__(self, 
                 xnli_data_path: str,
                 language: str,
                 results_dir: str = 'entailment_surfaces/surfaces_slurm/logs/xnli_phdim_results',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 seed: int = 42):
        """
        Initialize analyzer
        
        Args:
            xnli_data_path: Path to processed XNLI embeddings (.pt file)
            language: Language code (e.g., 'en', 'zh', 'ar')
            results_dir: Directory to save analysis results
            device: Computing device
            seed: Random seed
        """
        self.xnli_data_path = xnli_data_path
        self.language = language
        self.results_dir = Path(results_dir) / language
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.seed = seed

        # Set all random seeds immediately
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Only the metrics and spaces we need
        self.distance_metrics = ['braycurtis', 'cosine']
        self.embedding_spaces = ['sbert_concat', 'lattice_containment']

        # PH-Dim parameters (same for both H0 and H1)
        self.phd_params = {
            'min_points': 200,
            'max_points': 1000,
            'point_jump': 50,
            'alpha': 1.0,
            'seed': seed
        }

        print(f"XNLI PH-Dimension Analyzer initialized (H0 + H1)")
        print(f"Using robust ph_dim_and_diagrams_from_distance_matrix method")
        print(f"Language: {language}")
        print(f"Seed: {seed}")
        print(f"Device: {self.device}")
        print(f"XNLI data: {xnli_data_path}")
        print(f"Results directory: {self.results_dir}")
        print(f"Distance metrics: {self.distance_metrics}")
        print(f"Embedding spaces: {self.embedding_spaces}")
        print(f"Computing both H0 and H1 PH-Dimensions")
        flush_output()

        self._load_preprocessed_data()

    def _load_preprocessed_data(self):
        """Load pre-processed XNLI embeddings"""
        print(f"Loading XNLI data for {self.language}...")

        if not os.path.exists(self.xnli_data_path):
            raise FileNotFoundError(f"XNLI data not found: {self.xnli_data_path}")

        self.xnli_data = torch.load(self.xnli_data_path, map_location=self.device, weights_only=False)

        print(f"XNLI data loaded:")
        print(f"  Premise embeddings: {self.xnli_data['premise_embeddings'].shape}")
        print(f"  Hypothesis embeddings: {self.xnli_data['hypothesis_embeddings'].shape}")
        print(f"  Labels: {len(self.xnli_data['labels'])}")
        print(f"  Language: {self.xnli_data['metadata']['language']}")
        print(f"  Label distribution: {self.xnli_data['metadata']['label_counts']}")
        flush_output()

    def extract_embedding_spaces(self, max_samples_per_class: int = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract only the embedding spaces we need
        
        Args:
            max_samples_per_class: Limit samples per class (None for all)
            
        Returns:
            Dict mapping space names to class embeddings
        """
        print("Extracting embedding spaces...")
        
        # CRITICAL: Set seed again before any random operations
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        premise_embs = self.xnli_data['premise_embeddings']
        hypothesis_embs = self.xnli_data['hypothesis_embeddings']
        labels = self.xnli_data['labels']

        # Group by class
        data_by_class = {'entailment': {}, 'neutral': {}, 'contradiction': {}}

        for label in data_by_class.keys():
            mask = torch.tensor([l == label for l in labels], device=self.device, dtype=torch.bool)
            indices = torch.where(mask)[0]
            
            if max_samples_per_class and len(indices) > max_samples_per_class:
                # Seeded random sampling
                perm = torch.randperm(len(indices), device=self.device)[:max_samples_per_class]
                indices = indices[perm]
            
            data_by_class[label] = {
                'premise_bert': premise_embs[indices],
                'hypothesis_bert': hypothesis_embs[indices],
                'indices': indices
            }
            
            print(f"  {label}: {len(indices)} samples")

        # Extract embedding spaces
        all_embeddings = {}

        for space in self.embedding_spaces:
            print(f"Extracting {space}...")
            space_embeddings = {}
            
            for label in data_by_class.keys():
                if space == 'sbert_concat':
                    space_embeddings[label] = torch.cat([
                        data_by_class[label]['premise_bert'],
                        data_by_class[label]['hypothesis_bert']
                    ], dim=1).cpu()
                    
                elif space == 'lattice_containment':
                    epsilon = 1e-8
                    premise_bert = data_by_class[label]['premise_bert']
                    hypothesis_bert = data_by_class[label]['hypothesis_bert']
                    space_embeddings[label] = ((premise_bert * hypothesis_bert) / 
                                              (torch.abs(premise_bert) + torch.abs(hypothesis_bert) + epsilon)).cpu()

            all_embeddings[space] = space_embeddings

            # Print shapes
            for label, embs in space_embeddings.items():
                print(f"  {space} {label}: {embs.shape}")
                flush_output()

        return all_embeddings

    def compute_phdim_both_dims(self, embeddings: torch.Tensor, metric: str, 
                               class_name: str, space_name: str) -> Tuple[float, float, int, int]:
        """
        Compute BOTH H0 and H1 PH-Dim using robust method
        
        Args:
            embeddings: Class embeddings
            metric: Distance metric
            class_name: Entailment class name
            space_name: Embedding space name
            
        Returns:
            Tuple[float, float, int, int]: (h0_phdim, h1_phdim, h0_feature_count, h1_feature_count)
        """
        print(f"  Computing H0 and H1 PH-Dim for {class_name} in {space_name} using {metric}")
        flush_output()
        
        if len(embeddings) < self.phd_params['min_points']:
            print(f"    Warning: Only {len(embeddings)} samples, need ≥{self.phd_params['min_points']}")
            return np.nan, np.nan, 0, 0
        
        try:            
            # Sample if needed
            max_points = min(self.phd_params['max_points'], len(embeddings))
            if len(embeddings) > max_points:
                torch.manual_seed(self.phd_params['seed'])
                np.random.seed(self.phd_params['seed'])
                
                if embeddings.device.type == 'cpu':
                    indices = torch.randperm(len(embeddings))[:max_points]
                else:
                    indices = torch.randperm(len(embeddings), device=self.device)[:max_points]
                embeddings = embeddings[indices]

            embeddings_np = embeddings.detach().cpu().numpy()
            
            # Compute distance matrix
            print(f"    Computing distance matrix...")
            dm = pairwise_distances(embeddings_np, metric=metric)
            
            # Compute H0 PH-Dim
            print(f"    Computing H0 PH-Dim...")
            phd_h0, diagrams_h0 = ph_dim_and_diagrams_from_distance_matrix(
                dm,
                min_points=self.phd_params['min_points'],
                max_points=self.phd_params['max_points'],
                point_jump=self.phd_params['point_jump'],
                h_dim=0,
                alpha=self.phd_params['alpha'],
                seed=self.phd_params['seed']
            )
            
            # Count H0 features in first diagram
            h0_count = 0
            if len(diagrams_h0) > 0 and len(diagrams_h0[0]) > 0:
                h0_features = diagrams_h0[0][0]  # First sample's H0 diagram
                h0_count = len(h0_features[h0_features[:, 1] < np.inf])
            
            # Compute H1 PH-Dim
            print(f"    Computing H1 PH-Dim...")
            phd_h1, diagrams_h1 = ph_dim_and_diagrams_from_distance_matrix(
                dm,
                min_points=self.phd_params['min_points'],
                max_points=self.phd_params['max_points'],
                point_jump=self.phd_params['point_jump'],
                h_dim=1,
                alpha=self.phd_params['alpha'],
                seed=self.phd_params['seed']
            )
            
            # Count H1 features in first diagram
            h1_count = 0
            if len(diagrams_h1) > 0 and len(diagrams_h1[0]) > 1:
                h1_features = diagrams_h1[0][1]  # First sample's H1 diagram
                h1_count = len(h1_features[h1_features[:, 1] < np.inf])
            
            print(f"    {class_name} H0 PH-Dim: {phd_h0:.4f} ({h0_count} H0 features)")
            print(f"    {class_name} H1 PH-Dim: {phd_h1:.4f} ({h1_count} H1 features)")
            
            return phd_h0, phd_h1, h0_count, h1_count
            
        except Exception as e:
            print(f"    Error computing PH-Dim for {class_name}: {e}")
            import traceback
            traceback.print_exc()
            return np.nan, np.nan, 0, 0


    def run_analysis(self):
        """Run PH-Dimension analysis (H0 and H1) for current language and seed"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("="*80)
        print(f"PH-DIMENSION ANALYSIS (H0 & H1) - XNLI {self.language.upper()} (seed: {self.seed})")
        print(f"Device: {self.device}")
        print("="*80)
        
        # Extract embedding spaces
        all_embeddings = self.extract_embedding_spaces()
        
        # Results storage
        all_results = {}
        
        # Test each embedding space
        for space_name, space_embeddings in all_embeddings.items():
            print(f"\n{'='*60}")
            print(f"TESTING SPACE: {space_name}")
            print(f"{'='*60}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            space_results = {}
            
            # Test each distance metric
            for metric in self.distance_metrics:
                print(f"\n--- Testing {metric} metric ---")
                flush_output()
                
                metric_results = {}
                phd_h0_scores = {}
                phd_h1_scores = {}
                h0_feature_counts = {}
                h1_feature_counts = {}
                
                # Compute BOTH H0 and H1 PH-Dim for each class
                for class_name, embeddings in space_embeddings.items():
                    phd_h0, phd_h1, h0_count, h1_count = self.compute_phdim_both_dims(
                        embeddings, metric, class_name, space_name
                    )
                    phd_h0_scores[class_name] = phd_h0
                    phd_h1_scores[class_name] = phd_h1
                    h0_feature_counts[class_name] = h0_count
                    h1_feature_counts[class_name] = h1_count
                    metric_results[f'phd_h0_{class_name}'] = phd_h0
                    metric_results[f'phd_h1_{class_name}'] = phd_h1
                    metric_results[f'h0_features_{class_name}'] = h0_count
                    metric_results[f'h1_features_{class_name}'] = h1_count
                
                space_results[metric] = metric_results
                               
                print(f"  H0 PH-Dim scores: {phd_h0_scores}")                
                print(f"  H1 PH-Dim scores: {phd_h1_scores}")
                print(f"  H0 feature counts: {h0_feature_counts}")
                print(f"  H1 feature counts: {h1_feature_counts}")
            
            all_results[space_name] = space_results
        
        # Save results
        results_file = self.results_dir / f"phdim_{self.language}_seed{self.seed}_{timestamp}.json"
        results_with_metadata = {
            'language': self.language,
            'seed': self.seed,
            'timestamp': timestamp,
            'results': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        print(f"\nSaved results to {results_file}")
        
        # Generate simple report
        self._generate_report(all_results, timestamp)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}")
        
        return all_results

    def _generate_report(self, results: Dict, timestamp: str):
        """Generate plain text report with both H0 and H1"""
        report_file = self.results_dir / f"phdim_report_{self.language}_seed{self.seed}_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"PH-DIMENSION ANALYSIS REPORT (H0 & H1) - XNLI {self.language.upper()}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write("="*80 + "\n\n")
            
            for space_name, space_results in results.items():
                f.write(f"EMBEDDING SPACE: {space_name}\n")
                f.write("-" * 50 + "\n")
                
                for metric, metric_results in space_results.items():
                    f.write(f"\nMetric: {metric}\n")
                    
                    # H0 PH-Dim results
                    f.write("  H0 PH-Dim Analysis:\n")
                    for key, value in metric_results.items():
                        if key.startswith('phd_h0_'):
                            f.write(f"    {key}: {value}\n")
                    
                    # H1 PH-Dim results
                    f.write("  H1 PH-Dim Analysis:\n")
                    for key, value in metric_results.items():
                        if key.startswith('phd_h1_'):
                            f.write(f"    {key}: {value}\n")
                    
                    # Feature counts
                    f.write("  Feature Counts:\n")
                    for key, value in metric_results.items():
                        if key.startswith('h0_features_') or key.startswith('h1_features_'):
                            f.write(f"    {key}: {value}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Report saved to: {report_file}")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True,
                       help='Language code (e.g., en, zh, ar)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed (e.g., 42, 101, 333, 444, 500)')
    args = parser.parse_args()
    
    language = args.language
    seed = args.seed
    
    # Validate language
    all_languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    if language not in all_languages:
        print(f"ERROR: Unknown language '{language}'")
        print(f"Available languages: {', '.join(all_languages)}")
        return None
    
    # Validate seed
    recommended_seeds = [42, 101, 333, 444, 500]
    if seed not in recommended_seeds:
        print(f"WARNING: Seed {seed} not in recommended seeds: {recommended_seeds}")
    
    xnli_data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/processed/xnli_{language}_combined_SBERT.pt"
    
    if not Path(xnli_data_path).exists():
        print(f"ERROR: Data file not found: {xnli_data_path}")
        return None
    
    # Initialize analyzer
    analyzer = XNLIPHDimAnalyzer(
        xnli_data_path=xnli_data_path,
        language=language,
        results_dir="entailment_surfaces/surfaces_slurm/logs/xnli_phdim_results",
        seed=seed
    )
    
    # Run analysis
    results = analyzer.run_analysis()
    
    print(f"\nXNLI PH-Dimension analysis (H0 & H1) completed for {language} (seed {seed})!")
    return results


if __name__ == "__main__":
    main()