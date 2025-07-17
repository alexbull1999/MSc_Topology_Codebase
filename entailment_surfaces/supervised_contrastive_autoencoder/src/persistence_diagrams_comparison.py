"""
Analyze similarity of persistence diagrams within each class from the 100% clustering results.
This helps determine if the diagrams are similar enough to average into prototypes.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import sys
import os
from collections import defaultdict
from gph.python import ripser_parallel
from persim import PersistenceImager, bottleneck, wasserstein

# Add path to existing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phd_method', 'src_phd'))
from sklearn.metrics.pairwise import pairwise_distances

def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                           min_points=200,
                                           max_points=1000,
                                           point_jump=50,
                                           h_dim=0,
                                           alpha: float = 1.,
                                           seed: int = 42) -> Tuple[float, List[np.ndarray]]:
    """
    Compute both PH dimension and persistence diagrams from distance matrix
    Adapted from your existing function to return both
    """

    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape
    
    # np.random.seed(seed)
    
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



class PersistenceDiagramCollector:
    """
    Collect persistence diagrams using the same methodology as phdim_clustering_validation_best_metrics.py
    """
    
    def __init__(self, embedding_space='sbert_concat', distance_metric='cosine', 
                 bert_data_path=None, device='cuda'):
        self.embedding_space = embedding_space
        self.distance_metric = distance_metric
        self.bert_data_path = bert_data_path
        self.device = device
        self.bert_data = None
        
        # Parameters matching your successful clustering
        self.phd_params = {
            'min_points': 200,
            'max_points': 1000,
            'point_jump': 50,
            'h_dim': 0,
            'alpha': 1.0
        }
        
    def load_data(self):
        """Load BERT data same as your existing files"""
        if self.bert_data_path is None:
            # Use default path from your setup
            self.bert_data_path = 'data/processed/snli_full_standard_SBERT.pt'
        
        self.bert_data = torch.load(self.bert_data_path, map_location=self.device, weights_only=False)
        print(f"Loaded BERT data: {self.bert_data['premise_embeddings'].shape}")
        labels_list = self.bert_data['labels']
        unique_labels = sorted(list(set(labels_list))) # e.g., ['contradiction', 'entailment', 'neutral']
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        int_labels = [label_to_int[label] for label in labels_list]
        labels_tensor = torch.tensor(int_labels)
        print(f"Labels distribution: {torch.bincount(labels_tensor)}")
        print(f"Label mapping used for bincount: {label_to_int}") 

        
    def extract_class_embeddings(self, class_idx, n_points=1000):
        """
        Extract embeddings for a specific class
        
        Args:
            class_idx: 0=entailment, 1=neutral, 2=contradiction
            n_points: Number of points to sample
            use_current_random_state: If True, use current numpy random state instead of reseeding
        """
        if self.bert_data is None:
            self.load_data()
            
        # Get indices for this class
        if class_idx == 0:
            class_name = 'entailment'
        elif class_idx == 1:
            class_name = 'neutral'
        elif class_idx == 2:
            class_name = 'contradiction'
        else:
            raise ValueError 

        class_mask = torch.tensor([label == class_name for label in self.bert_data['labels']])
        class_indices = torch.where(class_mask)[0]
        
        # Sample n_points randomly
        if len(class_indices) > n_points:
            sampled_indices = np.random.choice(class_indices.cpu().numpy(), n_points, replace=False)
        else:
            sampled_indices = class_indices.cpu().numpy()
        
        # Extract embeddings based on embedding space
        if self.embedding_space == 'sbert_concat':
            premise_emb = self.bert_data['premise_embeddings'][sampled_indices]
            hypothesis_emb = self.bert_data['hypothesis_embeddings'][sampled_indices]
            embeddings = torch.cat([premise_emb, hypothesis_emb], dim=1)
        else:
            raise NotImplementedError(f"Embedding space {self.embedding_space} not implemented")
        
        return embeddings.detach().cpu().numpy()

    def collect_persistence_diagrams(self, n_tests=10, n_samples_per_test=10):
        """
        Collect persistence diagrams using the same methodology as your 100% clustering
        10 independent tests, each with 10 samples of 1000 points per class
        
        Following the exact seeding procedure from phdim_clustering_validation_best_metrics.py:
        - Each test run gets a different seed (42, 43, 44, ..., 51)
        - Within each test run, that same seed is used to select ALL samples for that run
        - The seed determines the sequence of random samples for E, N, C across all 10 samples
        """
        print(f"Collecting persistence diagrams for {self.embedding_space} + {self.distance_metric}")
        print(f"Running {n_tests} tests with {n_samples_per_test} samples each")
        print("Following exact seeding procedure from phdim_clustering_validation_best_metrics.py")
        
        class_names = ['entailment', 'neutral', 'contradiction']
        all_diagrams = {
            'entailment': {'H0': [], 'H1': [], 'phd_scores': []},
            'neutral': {'H0': [], 'H1': [], 'phd_scores': []},
            'contradiction': {'H0': [], 'H1': [], 'phd_scores': []}
        }
        
        for test_idx in range(n_tests):
            # Each test run gets its own seed (42, 43, 44, ..., 51)
            test_seed = 42 + test_idx
            print(f"\nTest {test_idx + 1}/{n_tests} (seed={test_seed})")
            
            # Set the seed once for this entire test run
            np.random.seed(test_seed)
            
            for sample_idx in range(n_samples_per_test):
                for class_idx, class_name in enumerate(class_names):
                    # Extract embeddings using the current random state (no re-seeding)
                    embeddings = self.extract_class_embeddings(
                        class_idx, n_points=1000
                    )
                    
                    # Compute distance matrix
                    distance_matrix = pairwise_distances(embeddings, metric=self.distance_metric)
                    
                    # Compute persistence diagrams with fixed seed=42 for reproducible distance matrix calculations
                    phd_score, diagrams = ph_dim_and_diagrams_from_distance_matrix(
                        distance_matrix,
                        min_points=self.phd_params['min_points'],
                        max_points=self.phd_params['max_points'],
                        point_jump=self.phd_params['point_jump'],
                        h_dim=self.phd_params['h_dim'],
                        alpha=self.phd_params['alpha']
                        # seed=42  
                    )
                    
                    # Store results
                    all_diagrams[class_name]['H0'].append(diagrams[0])
                    all_diagrams[class_name]['H1'].append(diagrams[1])
                    all_diagrams[class_name]['phd_scores'].append(phd_score)
                    
                    print(f"  Sample {sample_idx + 1}, {class_name}: H0={len(diagrams[0])}, H1={len(diagrams[1])}, PHD={phd_score:.2f}")
        
        return all_diagrams


class PersistenceDiagramSimilarityAnalyzer:
    """
    Performs comprehensive similarity analysis of persistence diagrams using robust methods.

    Key Improvements:
    1.  **Robust Persistence Images**: Uses the `persim` library for correct and
        standardized persistence image generation, as requested.
    2.  **Correct Distance Metrics**: Replaced naive implementations of Bottleneck and
        Wasserstein distances with the industry-standard, accurate algorithms from `persim`.
    3.  **Configurable & Standardized**: PersistenceImager is now a configurable
        class attribute for easier tuning and consistent results.
    4.  **Clearer Analysis**: The analysis loop is more streamlined and focuses on
        the most meaningful metrics.
    """
    def __init__(self, diagrams_data: Dict):
        self.diagrams_data = diagrams_data
        # Initialize a standardized PersistenceImager based on your working code
        self.pimgr = PersistenceImager(
            pixel_size=0.5,
            birth_range=(0, 5),
            pers_range=(0, 5),
            kernel_params={'sigma': 0.3}
        )
        print("Initialized Analyzer with robust `persim` methods for images and distances.")


    def _get_persistence_statistics(self, diagram: np.ndarray) -> Dict:
        """
        Extracts a richer set of statistics from a single persistence diagram.
        """
        if diagram.size == 0:
            return {
                'total_persistence': 0.0, 'max_persistence': 0.0, 'mean_persistence': 0.0,
                'feature_count': 0, 'birth_mean': 0.0, 'birth_std': 0.0,
                'death_mean': 0.0, 'death_std': 0.0
            }

        finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
        if finite_diagram.size == 0:
            return {
                'total_persistence': 0.0, 'max_persistence': 0.0, 'mean_persistence': 0.0,
                'feature_count': 0, 'birth_mean': 0.0, 'birth_std': 0.0,
                'death_mean': 0.0, 'death_std': 0.0
            }

        persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
        births = finite_diagram[:, 0]
        deaths = finite_diagram[:, 1]

        return {
            'total_persistence': np.sum(persistences),
            'max_persistence': np.max(persistences),
            'mean_persistence': np.mean(persistences),
            'feature_count': len(finite_diagram),
            'birth_mean': np.mean(births),
            'birth_std': np.std(births),
            'death_mean': np.mean(deaths),
            'death_std': np.std(deaths)
        }


    def _diagram_to_image(self, diagram: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts a single persistence diagram to a standardized persistence image.
        This is the improved method using `persim`.
        """
        if diagram.size == 0:
            return None # Return None to signify no image could be made

        # Filter for finite points, as persim requires it
        finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
        if finite_diagram.size == 0:
            return None

        try:
            # transform expects a list of diagrams, so we wrap it
            img_list = self.pimgr.transform([finite_diagram])
            img = img_list[0].copy()  # Extract the first (and only) image
            del img_list
            return img
        except Exception as e:
            print(f"Warning: Could not create persistence image. Error: {e}")
            return None

    
    def _compute_image_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Computes the Pearson correlation between two flattened persistence images."""
        if img1 is None or img2 is None:
            return np.nan # Cannot compute if one image is missing

        flat1 = img1.flatten()
        flat2 = img2.flatten()

        # Correlation is undefined if one of the images is constant (std dev is zero)
        if np.std(flat1) == 0 or np.std(flat2) == 0:
            return 0.0 # Or np.nan, depending on desired handling

        corr = np.corrcoef(flat1, flat2)[0, 1]
        del flat1, flat2
        return corr if not np.isnan(corr) else 0.0

    
    def analyze_similarities(self, h_dim: int = 0) -> Dict:
        """
        Runs the full similarity analysis for a given homology dimension (0 or 1).
        Returns a dictionary with computed distances, correlations, and aggregated statistics.
        """
        analysis_results = {}
        dim_label = f'H{h_dim}'
        print(f"\n{'='*30}\nAnalyzing Similarity for {dim_label}\n{'='*30}")

        for class_name, data in self.diagrams_data.items():
            print(f"\n--- Processing Class: {class_name.upper()} ---")
            diagrams = data[dim_label]
            n_diagrams = len(diagrams)
            if n_diagrams < 2:
                print("  Not enough diagrams to compare.")
                continue

            # --- 1. Compute Pairwise Distances & Correlations ---
            bottleneck_dists, wasserstein_dists, image_corrs = [], [], []
            images = [self._diagram_to_image(d) for d in diagrams]

            for i in range(n_diagrams):
                for j in range(i + 1, n_diagrams):
                    b_dist = bottleneck(diagrams[i], diagrams[j], matching=False)
                    w_dist = wasserstein(diagrams[i], diagrams[j])
                    corr = self._compute_image_correlation(images[i], images[j])

                    bottleneck_dists.append(b_dist)
                    wasserstein_dists.append(w_dist)
                    if not np.isnan(corr): image_corrs.append(corr)

            # --- 2. Aggregate Statistics Across All Diagrams in the Class ---
            all_stats = [self._get_persistence_statistics(d) for d in diagrams]
            aggregated_stats = {}
            if all_stats:
                stat_keys = all_stats[0].keys()
                for key in stat_keys:
                    values = [s[key] for s in all_stats]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    # Coefficient of Variation (CV): A normalized measure of dispersion
                    cv = (std_val / mean_val) if mean_val != 0 else 0
                    aggregated_stats[key] = {'mean': mean_val, 'std': std_val, 'cv': cv}

            analysis_results[class_name] = {
                'bottleneck': {'mean': np.mean(bottleneck_dists), 'std': np.std(bottleneck_dists)},
                'wasserstein': {'mean': np.mean(wasserstein_dists), 'std': np.std(wasserstein_dists)},
                'image_correlation': {'mean': np.mean(image_corrs), 'std': np.std(image_corrs)},
                'statistics': aggregated_stats # Store the aggregated stats
            }
            print(f"  Image Correlation: {analysis_results[class_name]['image_correlation']['mean']:.4f} ± {analysis_results[class_name]['image_correlation']['std']:.4f}")
            print(f"  Stat [Total Persistence] CV: {aggregated_stats.get('total_persistence',{}).get('cv', 0):.3f}")


        return analysis_results


    def plot_analysis(self, h0_results: Dict, h1_results: Dict, save_path: str = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/similarity_analysis.png'):
        """Visualizes the similarity analysis results."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15), dpi=100)
        fig.suptitle('Persistence Diagram Similarity Analysis (Within-Class)', fontsize=16, y=1.02)
        class_names = list(h0_results.keys())

        metrics = ['image_correlation', 'bottleneck', 'wasserstein']
        metric_titles = ['Persistence Image Correlation', 'Bottleneck Distance', 'Wasserstein Distance']

        for row, (metric, title) in enumerate(zip(metrics, metric_titles)):
            for col, class_name in enumerate(class_names):
                ax = axes[row, col]
                h0_vals = h0_results.get(class_name, {}).get(metric, {}).get('values', [])
                h1_vals = h1_results.get(class_name, {}).get(metric, {}).get('values', [])

                if h0_vals:
                    sns.histplot(h0_vals, ax=ax, color='skyblue', label='H0', kde=True, stat="density")
                if h1_vals:
                    sns.histplot(h1_vals, ax=ax, color='salmon', label='H1', kde=True, stat="density")

                ax.set_title(f'{class_name.title()}')
                ax.set_xlabel(title)
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path)
        print(f"Analysis plots saved to {save_path}")
        plt.show()

    def generate_report(self, h0_results: Dict, h1_results: Dict) -> str:
        """
        Generates a comprehensive text report with the new cross-class statistical comparison.
        """
        report = ["="*80, "COMPREHENSIVE PERSISTENCE DIAGRAM SIMILARITY REPORT", "="*80]
        report.append("This report assesses if persistence diagrams within each class are similar enough to be averaged into a 'prototype' for regularization.\n")
        report.append("Low CV (Coefficient of Variation) indicates high stability and suitability for averaging.")
        report.append("High Image Correlation and Low Bottleneck/Wasserstein Distance are desirable.")

        # --- Section 1: Per-Class Summary ---
        report.append("\n" + "="*40 + " PER-CLASS SIMILARITY SUMMARY " + "="*40)
        for class_name in self.diagrams_data.keys():
            report.append(f"\n--- {class_name.upper()} CLASS ---")
            h0_corr = h0_results.get(class_name, {}).get('image_correlation', {'mean': np.nan})['mean']
            h1_corr = h1_results.get(class_name, {}).get('image_correlation', {'mean': np.nan})['mean']
            h0_bneck = h0_results.get(class_name, {}).get('bottleneck', {'mean': np.nan})['mean']
            h1_bneck = h1_results.get(class_name, {}).get('bottleneck', {'mean': np.nan})['mean']

            report.append(f"  H0: Avg. Image Correlation = {h0_corr:.3f}, Avg. Bottleneck Dist = {h0_bneck:.4f}")
            report.append(f"  H1: Avg. Image Correlation = {h1_corr:.3f}, Avg. Bottleneck Dist = {h1_bneck:.4f}")

        # --- Section 2: NEW Cross-Class Statistical Profile ---
        report.append("\n\n" + "="*40 + " CROSS-CLASS STATISTICAL PROFILE " + "="*40)
        stat_keys_to_report = ['total_persistence', 'max_persistence', 'feature_count']

        for h_dim, results in [('H0', h0_results), ('H1', h1_results)]:
            report.append(f"\n--- {h_dim} STATISTICS ---")
            for stat_key in stat_keys_to_report:
                report.append(f"\n  Metric: {stat_key.replace('_', ' ').title()}")
                header = f"    {'CLASS':<15} | {'MEAN':>15} | {'STD DEV':>15} | {'CV (STABILITY)':>18}"
                report.append(header)
                report.append("    " + "-"*len(header))
                for class_name in self.diagrams_data.keys():
                    stats = results.get(class_name, {}).get('statistics', {}).get(stat_key, {})
                    mean_val = stats.get('mean', np.nan)
                    std_val = stats.get('std', np.nan)
                    cv_val = stats.get('cv', np.nan)
                    report.append(f"    {class_name:<15} | {mean_val:>15.2f} | {std_val:>15.2f} | {cv_val:>18.3f}")

        # --- Section 3: Final Assessment ---
        report.append("\n\n" + "="*40 + " FINAL ASSESSMENT & RECOMMENDATION " + "="*40)
        for class_name in self.diagrams_data.keys():
            report.append(f"\n--- {class_name.upper()} CLASS ---")
            h1_corr = h1_results.get(class_name, {}).get('image_correlation', {'mean': 0})['mean']
            h1_stats = h1_results.get(class_name, {}).get('statistics', {})
            h1_total_pers_cv = h1_stats.get('total_persistence', {'cv': 999}).get('cv', 999)

            report.append(f"  Topological Stability (H1 Total Persistence CV): {h1_total_pers_cv:.3f}")
            report.append(f"  Structural Similarity (H1 Image Correlation):   {h1_corr:.3f}")

            if h1_total_pers_cv < 0.25 and h1_corr > 0.6:
                assessment = "EXCELLENT: Very high stability and similarity. Ideal for averaging."
            elif h1_total_pers_cv < 0.5 and h1_corr > 0.4:
                assessment = "GOOD: Solid stability and similarity. Averaging is recommended and likely to be effective."
            elif h1_total_pers_cv < 0.75 or h1_corr > 0.25:
                assessment = "FAIR: Moderate stability. Averaging may work but prototype might not capture all variance. Proceed with caution."
            else:
                assessment = "POOR: High variance and low similarity. Averaging is not recommended as the 'average' diagram will likely not be representative."
            report.append(f"  Recommendation: {assessment}")

        return "\n".join(report)


if __name__ == '__main__':
    # 1. Collect the persistence diagrams
    SAVE_PATH = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/collected_diagrams.pkl'
    if Path(SAVE_PATH).exists():
        print(f"Loading pre-computed diagrams from {SAVE_PATH}...")
        with open(SAVE_PATH, 'rb') as f:
            all_diagrams = pickle.load(f)
    else:
        print("Running diagram collection...")
        collector = PersistenceDiagramCollector(bert_data_path='data/processed/snli_full_standard_SBERT.pt')
        all_diagrams = collector.collect_persistence_diagrams(n_tests=10, n_samples_per_test=10)
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(all_diagrams, f)
        print(f"Diagrams collected and saved to {SAVE_PATH}")


    # 2. Initialize the analyzer with the collected data
    analyzer = PersistenceDiagramSimilarityAnalyzer(all_diagrams)

    # 3. Run the analysis for both H0 and H1
    h0_analysis_results = analyzer.analyze_similarities(h_dim=0)
    h1_analysis_results = analyzer.analyze_similarities(h_dim=1)

    # 4. Generate the plots (This part of the code is unchanged)
    # analyzer.plot_analysis(h0_analysis_results, h1_analysis_results)

    # 5. Generate and print the final, enhanced report
    report = analyzer.generate_report(h0_analysis_results, h1_analysis_results)
    print("\n\n" + report)

    REPORT_SAVE_PATH = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/similarity_report.txt'
    with open(REPORT_SAVE_PATH, 'w') as f:
        f.write(report)
    print(f"\nFinal report successfully saved to: {REPORT_SAVE_PATH}")


