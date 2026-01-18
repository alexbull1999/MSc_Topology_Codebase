import torch
import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pacmap
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_embeddings(file_path):
    """Load the encoded contradiction embeddings"""
    print(f"Loading embeddings from: {file_path}")
    data = torch.load(file_path, weights_only=False)
    
    premise_embeddings = data['premise_embeddings']
    hypothesis_embeddings = data['hypothesis_embeddings']
    
    print(f"Loaded {len(premise_embeddings)} contradiction samples")
    print(f"Embedding dimension: {premise_embeddings.shape[1]}")
    
    return premise_embeddings, hypothesis_embeddings

def create_embedding_spaces(premise_embeddings, hypothesis_embeddings):
    """Create different embedding space representations
    
    Returns:
        dict: Dictionary with embedding space names as keys and arrays as values
    """
    spaces = {}
    
    # 1. Simple Concatenation (most successful in thesis)
    spaces['concatenation'] = np.concatenate([premise_embeddings, hypothesis_embeddings], axis=1)
    print(f"Concatenation space shape: {spaces['concatenation'].shape}")
    
    # 2. Semantic Coherence (lattice containment - also successful in thesis)
    # Formula: (p * h) / (|p| + |h| + epsilon)
    epsilon = 1e-8
    element_product = premise_embeddings * hypothesis_embeddings
    premise_norm = np.linalg.norm(premise_embeddings, axis=1, keepdims=True)
    hypothesis_norm = np.linalg.norm(hypothesis_embeddings, axis=1, keepdims=True)
    spaces['semantic_coherence'] = element_product / (premise_norm + hypothesis_norm + epsilon)
    print(f"Semantic Coherence space shape: {spaces['semantic_coherence'].shape}")
    
    return spaces

def run_clustering(embeddings, n_clusters, random_state=42):
    """Run multiple clustering algorithms
    
    Args:
        embeddings: Input embeddings (N x D)
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        dict: Dictionary with algorithm names and cluster labels
    """
    results = {}
    
    print(f"\nRunning clustering with k={n_clusters}...")
    
    # Spectral Clustering
    print("  - Spectral Clustering...")
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity='nearest_neighbors')
    results['spectral'] = spectral.fit_predict(embeddings)
    
    # Gaussian Mixture Model
    print("  - Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    results['gmm'] = gmm.fit_predict(embeddings)
    
    # Hierarchical Clustering
    print("  - Hierarchical Clustering...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    results['hierarchical'] = hierarchical.fit_predict(embeddings)
    
    return results

def compute_metrics(embeddings, labels):
    """Compute clustering evaluation metrics
    
    Args:
        embeddings: Input embeddings
        labels: Cluster labels
        
    Returns:
        dict: Dictionary with metric names and values
    """
    metrics = {}
    
    # Silhouette Score
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    else:
        metrics['silhouette'] = -1.0
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
    metrics['n_clusters'] = len(unique)
    
    return metrics

def create_pacmap_visualization(embeddings, labels, algorithm_name, embedding_space, k, output_dir):
    """Create PaCMAP 2D visualization of clusters
    
    Args:
        embeddings: Input embeddings
        labels: Cluster labels
        algorithm_name: Name of clustering algorithm
        embedding_space: Name of embedding space
        k: Number of clusters
        output_dir: Directory to save plots
    """
    print(f"    Creating PaCMAP visualization...")
    
    # Reduce to 2D with PaCMAP
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'{algorithm_name} - {embedding_space} (k={k})')
    plt.xlabel('PaCMAP Dimension 1')
    plt.ylabel('PaCMAP Dimension 2')
    plt.tight_layout()
    
    # Save plot
    filename = f'{embedding_space}_{algorithm_name}_k{k}_pacmap.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")

def main():
    # Configuration
    input_path = "/vol/bitbucket/ahb24/tda_entailment_new/contradictions_only/contradiction_embeddings_SBERT_snli_10k_subset_balanced.pt"
    output_dir = Path("contradiction_analysis/clustering_results_10k_subset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Number of clusters to test
    k_values = [2, 3, 4, 5, 10]
    
    print("=" * 80)
    print("CONTRADICTION CLUSTERING ANALYSIS")
    print("=" * 80)
    
    # Load embeddings
    premise_embeddings, hypothesis_embeddings = load_embeddings(input_path)
    
    # Create embedding spaces
    print("\n" + "=" * 80)
    print("CREATING EMBEDDING SPACES")
    print("=" * 80)
    embedding_spaces = create_embedding_spaces(premise_embeddings, hypothesis_embeddings)
    
    # Store all results
    all_results = {}
    
    # Run clustering for each embedding space and k value
    print("\n" + "=" * 80)
    print("RUNNING CLUSTERING EXPERIMENTS")
    print("=" * 80)
    
    for space_name, embeddings in embedding_spaces.items():
        print(f"\n{space_name.upper()} EMBEDDING SPACE")
        print("-" * 80)
        
        all_results[space_name] = {}
        
        for k in k_values:
            print(f"\nk={k}")
            all_results[space_name][k] = {}
            
            # Run clustering algorithms
            cluster_results = run_clustering(embeddings, k)
            
            # Compute metrics for each algorithm
            for algo_name, labels in cluster_results.items():
                print(f"  {algo_name}:")
                
                metrics = compute_metrics(embeddings, labels)
                all_results[space_name][k][algo_name] = metrics
                
                print(f"    Silhouette Score: {metrics['silhouette']:.4f}")
                print(f"    Cluster Sizes: {metrics['cluster_sizes']}")
                
                # Create PaCMAP visualization
                create_pacmap_visualization(embeddings, labels, algo_name, space_name, k, output_dir)
    
    # Save all results to JSON
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results_file = output_dir / "clustering_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for space_name in embedding_spaces.keys():
        print(f"\n{space_name.upper()}:")
        print("-" * 40)
        
        for k in k_values:
            print(f"\nk={k}:")
            for algo_name in ['spectral', 'gmm', 'hierarchical']:
                silhouette = all_results[space_name][k][algo_name]['silhouette']
                print(f"  {algo_name:15s}: silhouette = {silhouette:7.4f}")
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()