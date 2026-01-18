"""
PH-Dimension Analysis for XNLI Multilingual Data
Tests Bray-Curtis and Cosine on sbert_concat and lattice_containment across multiple seeds
"""

import torch
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phd_method.src_phd.topology import ph_dim_from_distance_matrix, fast_ripser

def flush_output():
    """Force output to appear immediately in SLURM"""
    sys.stdout.flush()
    sys.stderr.flush()

class XNLIPHDimAnalyzer:
    """
    PH-Dimension analysis for XNLI multilingual data
    Tests only Bray-Curtis and Cosine distance on sbert_concat and lattice_containment
    """
    
    def __init__(self, 
                 xnli_data_path: str,
                 language: str,
                 results_dir: str = '/vol/bitbucket/ahb24/tda_entailment_new/xnli_results',
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
        self.results_dir = Path(results_dir) / language / 'phdim_analysis'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        # Only the metrics and spaces we need
        self.distance_metrics = ['braycurtis', 'cosine']
        self.embedding_spaces = ['sbert_concat', 'lattice_containment']

        # PH-Dim parameters
        self.phd_params = {
            'min_points': 200,
            'max_points': 1000,
            'point_jump': 50,
            'h_dim': 0,
            'alpha': 1.0,
            'seed': seed
        }

        print(f"XNLI PH-Dimension Analyzer initialized for: {language} (seed: {seed})")
        print(f"Device: {self.device}")
        print(f"XNLI data: {xnli_data_path}")
        print(f"Distance metrics: {self.distance_metrics}")
        print(f"Embedding spaces: {self.embedding_spaces}")
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

        premise_embs = self.xnli_data['premise_embeddings']
        hypothesis_embs = self.xnli_data['hypothesis_embeddings']
        labels = self.xnli_data['labels']

        # Group by class
        data_by_class = {'entailment': {}, 'neutral': {}, 'contradiction': {}}

        for label in data_by_class.keys():
            mask = torch.tensor([l == label for l in labels], device=self.device, dtype=torch.bool)
            indices = torch.where(mask)[0]
            
            if max_samples_per_class and len(indices) > max_samples_per_class:
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

    def compute_distance_matrix(self, embeddings: torch.Tensor, metric: str) -> np.ndarray:
        """
        Compute distance matrix
        
        Args:
            embeddings: Embedding tensor [n_samples, embed_dim]
            metric: Distance metric name
            
        Returns:
            Distance matrix [n_samples, n_samples]
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        return pairwise_distances(embeddings_np, metric=metric)

    def compute_phdim(self, embeddings: torch.Tensor, metric: str, 
                     class_name: str, space_name: str) -> float:
        """
        Compute PH-Dim
        
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
            max_points = min(self.phd_params['max_points'], len(embeddings))
            if len(embeddings) > max_points:
                torch.manual_seed(self.phd_params['seed'])
                if embeddings.device.type == 'cpu':
                    indices = torch.randperm(len(embeddings))[:max_points]
                else:
                    indices = torch.randperm(len(embeddings), device=self.device)[:max_points]
                embeddings = embeddings[indices]

            embeddings_np = embeddings.detach().cpu().numpy()
            
            # Use fast_ripser for sklearn metrics
            if metric in ['braycurtis', 'cosine']:
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

    def compute_cross_class_analysis(self, embeddings_by_class: Dict[str, torch.Tensor], 
                                    metric: str, space_name: str) -> Dict[str, float]:
        """
        Analyze cross-class distances
        
        Args:
            embeddings_by_class: Embeddings organized by entailment class
            metric: Distance metric to test
            space_name: Embedding space name
            
        Returns:
            Dictionary with cross-class distance metrics
        """
        print(f"  Computing cross-class analysis for {space_name} using {metric}")
        flush_output()
        
        required_classes = {'entailment', 'neutral', 'contradiction'}
        available_classes = set(embeddings_by_class.keys())
        
        if not required_classes.issubset(available_classes):
            print(f"    Missing classes: {required_classes - available_classes}")
            return {}
        
        try:
            # Compute centroids
            centroids_gpu = {}
            for label, embeddings in embeddings_by_class.items():
                centroids_gpu[label] = torch.mean(embeddings, dim=0)
            
            # Centroid distances
            centroid_distances = {}
            for label1 in required_classes:
                for label2 in required_classes:
                    if label1 != label2:
                        c1, c2 = centroids_gpu[label1], centroids_gpu[label2]
                        
                        if metric == 'cosine':
                            dist = (1 - torch.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0))).item()
                        elif metric == 'braycurtis':
                            c1_np, c2_np = c1.cpu().numpy(), c2.cpu().numpy()
                            dist = pairwise_distances([c1_np], [c2_np], metric=metric)[0, 0]
                        else:
                            dist = torch.norm(c1 - c2).item()
                        
                        centroid_distances[f'{label1}_to_{label2}'] = dist
            
            # Minimum cross-class distances
            min_distances = {}
            
            for label1 in required_classes:
                for label2 in required_classes:
                    if label1 != label2:
                        sample_size = min(500, len(embeddings_by_class[label1]), len(embeddings_by_class[label2]))
                        
                        if sample_size < 10:
                            continue
                        
                        embs1 = embeddings_by_class[label1]
                        embs2 = embeddings_by_class[label2]
                        torch.manual_seed(self.phd_params['seed'])

                        if embs1.device.type == 'cpu':
                            idx1 = torch.randperm(len(embs1))[:sample_size]
                            idx2 = torch.randperm(len(embs2))[:sample_size]
                        else:
                            idx1 = torch.randperm(len(embs1), device=embs1.device)[:sample_size]
                            idx2 = torch.randperm(len(embs2), device=embs2.device)[:sample_size]
                        
                        embs1_sample = embs1[idx1].cpu().numpy()
                        embs2_sample = embs2[idx2].cpu().numpy()
                        
                        cross_distances = pairwise_distances(embs1_sample, embs2_sample, metric=metric)
                        min_distances[f'{label1}_to_{label2}'] = np.min(cross_distances)
            
            # Entailment separation
            entailment_separation = 0.0
            if 'entailment_to_neutral' in centroid_distances and 'entailment_to_contradiction' in centroid_distances:
                avg_entailment_distance = (centroid_distances['entailment_to_neutral'] + 
                                         centroid_distances['entailment_to_contradiction']) / 2
                
                entailment_embs = embeddings_by_class['entailment']
                entailment_centroid = centroids_gpu['entailment']
                entailment_spread = torch.std(torch.norm(entailment_embs - entailment_centroid, dim=1)).item()
                
                if entailment_spread > 0:
                    entailment_separation = avg_entailment_distance / entailment_spread
            
            results = {
                'centroid_ent_to_neutral': centroid_distances.get('entailment_to_neutral', 0),
                'centroid_ent_to_contradiction': centroid_distances.get('entailment_to_contradiction', 0),
                'centroid_neutral_to_contradiction': centroid_distances.get('neutral_to_contradiction', 0),
                'min_ent_to_neutral': min_distances.get('entailment_to_neutral', 0),
                'min_ent_to_contradiction': min_distances.get('entailment_to_contradiction', 0),
                'entailment_separation': entailment_separation,
            }
                        
            return results
            
        except Exception as e:
            print(f"    Error in cross-class analysis: {e}")
            return {}

    def run_analysis(self, timestamp: str):
        """Run PH-Dimension analysis for current seed"""
        print("="*80)
        print(f"PH-DIMENSION ANALYSIS - XNLI {self.language.upper()} (seed: {self.seed})")
        print(f"Device: {self.device}")
        print("="*80)
        
        # Extract embedding spaces (only once per seed)
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
                phd_scores = {}
                
                # Compute PH-Dim for each class
                for class_name, embeddings in space_embeddings.items():
                    phd = self.compute_phdim(embeddings, metric, class_name, space_name)
                    phd_scores[class_name] = phd
                    metric_results[f'phd_{class_name}'] = phd
                
                # Compute cross-class distances
                cross_class_metrics = self.compute_cross_class_analysis(
                    space_embeddings, metric, space_name
                )
                metric_results.update(cross_class_metrics)
                
                space_results[metric] = metric_results
                
                print(f"  PH-Dim scores: {phd_scores}")                
            
            all_results[space_name] = space_results
        
        # Save results for this seed
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
        
        return all_results


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='all',
                       help='Language code (e.g., en, zh, ar) or "all" for all languages')
    args = parser.parse_args()
    
    # All XNLI languages
    all_languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    
    # All seeds to test
    all_seeds = [42, 101, 333, 444, 500]
    
    # Determine which languages to process
    if args.language == 'all':
        languages_to_process = all_languages
        print(f"Processing all {len(all_languages)} languages")
    else:
        if args.language not in all_languages:
            print(f"ERROR: Unknown language '{args.language}'")
            print(f"Available languages: {', '.join(all_languages)}")
            return
        languages_to_process = [args.language]
        print(f"Processing single language: {args.language}")
    
    # Timestamp for this entire run
    global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each language
    successful = 0
    failed = 0
    
    for lang_idx, language in enumerate(languages_to_process, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING LANGUAGE {lang_idx}/{len(languages_to_process)}: {language.upper()}")
        print(f"{'='*80}")
        
        # Check if data exists
        xnli_data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/processed/xnli_{language}_combined_SBERT.pt"
        
        if not Path(xnli_data_path).exists():
            print(f"ERROR: Data file not found: {xnli_data_path}")
            failed += 1
            continue
        
        # Process each seed for this language
        language_successful = 0
        language_failed = 0
        
        for seed_idx, seed in enumerate(all_seeds, 1):
            print(f"\n{'-'*60}")
            print(f"SEED {seed_idx}/{len(all_seeds)}: {seed}")
            print(f"{'-'*60}")
            
            try:
                # Initialize analyzer with this seed
                analyzer = XNLIPHDimAnalyzer(
                    xnli_data_path=xnli_data_path,
                    language=language,
                    results_dir="/vol/bitbucket/ahb24/tda_entailment_new/xnli_results/phdim_values",
                    seed=seed
                )
                
                # Run analysis
                analyzer.run_analysis(global_timestamp)
                
                print(f"\n✓ {language.upper()} - seed {seed} complete!")
                language_successful += 1
                
                # Clear cache between seeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\n✗ {language.upper()} - seed {seed} FAILED!")
                print(f"Error: {e}")
                language_failed += 1
                
                # Clear cache even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Summary for this language
        print(f"\n{language.upper()} Summary:")
        print(f"  Successful seeds: {language_successful}/{len(all_seeds)}")
        print(f"  Failed seeds: {language_failed}/{len(all_seeds)}")
        
        if language_failed == 0:
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successful languages: {successful}/{len(languages_to_process)}")
    print(f"Failed languages: {failed}/{len(languages_to_process)}")
    print(f"Total result files created: {successful * len(all_seeds)}")
    
    if failed > 0:
        print("\n⚠ Some languages failed to process")
        exit(1)
    else:
        print("\n✓ All languages processed successfully!")
        exit(0)


if __name__ == "__main__":
    main()