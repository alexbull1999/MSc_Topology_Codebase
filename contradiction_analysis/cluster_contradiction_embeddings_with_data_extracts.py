import torch
import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pacmap
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd

def load_embeddings(file_path):
    """Load the encoded contradiction embeddings"""
    print(f"Loading embeddings from: {file_path}")
    data = torch.load(file_path, weights_only=False)
    
    premise_embeddings = data['premise_embeddings']
    hypothesis_embeddings = data['hypothesis_embeddings']
    metadata = data.get('metadata', None)
    
    print(f"Loaded {len(premise_embeddings)} contradiction samples")
    print(f"Embedding dimension: {premise_embeddings.shape[1]}")
    
    return premise_embeddings, hypothesis_embeddings, metadata

def create_embedding_spaces(premise_embeddings, hypothesis_embeddings):
    """Create different embedding space representations"""
    spaces = {}
    
    # Simple Concatenation
    spaces['concatenation'] = np.concatenate([premise_embeddings, hypothesis_embeddings], axis=1)
    print(f"Concatenation space shape: {spaces['concatenation'].shape}")
    
    # Semantic Coherence
    epsilon = 1e-8
    element_product = premise_embeddings * hypothesis_embeddings
    premise_norm = np.linalg.norm(premise_embeddings, axis=1, keepdims=True)
    hypothesis_norm = np.linalg.norm(hypothesis_embeddings, axis=1, keepdims=True)
    spaces['semantic_coherence'] = element_product / (premise_norm + hypothesis_norm + epsilon)
    print(f"Semantic Coherence space shape: {spaces['semantic_coherence'].shape}")
    
    return spaces

def run_clustering(embeddings, n_clusters, random_state=42):
    """Run multiple clustering algorithms"""
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

def create_pacmap_visualization(embeddings, labels, algorithm_name, embedding_space, k, output_dir):
    """Create PaCMAP 2D visualization of clusters"""
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

def save_raw_contradiction_data(metadata, premise_embeddings, hypothesis_embeddings, output_dir):
    """Save raw contradiction data for supervisor review"""
    print("\nSaving raw contradiction data...")
    
    # Create DataFrame with text and embeddings
    data_records = []
    for i, item in enumerate(metadata):
        record = {
            'index': i,
            'premise': item['premise'],
            'hypothesis': item['hypothesis'],
            'original_index': item.get('original_index', i)
        }
        data_records.append(record)
    
    df = pd.DataFrame(data_records)
    
    # Save text data to CSV
    csv_path = output_dir / 'contradiction_texts.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Text data saved to: {csv_path}")
    
    # Save embeddings separately (more efficient)
    embeddings_path = output_dir / 'contradiction_embeddings.npz'
    np.savez(embeddings_path, 
             premise_embeddings=premise_embeddings,
             hypothesis_embeddings=hypothesis_embeddings)
    print(f"  Embeddings saved to: {embeddings_path}")
    
    # Save combined small sample for easy inspection
    sample_size = min(100, len(metadata))
    sample_data = []
    for i in range(sample_size):
        sample_data.append({
            'index': i,
            'premise': metadata[i]['premise'],
            'hypothesis': metadata[i]['hypothesis']
        })
    
    sample_df = pd.DataFrame(sample_data)
    sample_path = output_dir / 'contradiction_sample_100.csv'
    sample_df.to_csv(sample_path, index=False)
    print(f"  Sample (100 examples) saved to: {sample_path}")

def save_cluster_assignments(all_cluster_assignments, metadata, output_dir):
    """Save cluster assignments for each algorithm/space combination"""
    print("\nSaving cluster assignments...")
    
    # Create base DataFrame
    df = pd.DataFrame([
        {
            'index': i,
            'premise': item['premise'],
            'hypothesis': item['hypothesis']
        }
        for i, item in enumerate(metadata)
    ])
    
    # Add cluster assignments for each configuration
    for space_name, space_results in all_cluster_assignments.items():
        for k, k_results in space_results.items():
            for algo_name, labels in k_results.items():
                col_name = f'{space_name}_{algo_name}_k{k}'
                df[col_name] = labels
    
    # Save to CSV
    csv_path = output_dir / 'cluster_assignments.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Cluster assignments saved to: {csv_path}")

def print_cluster_statistics(labels, algo_name, k):
    """Print cluster size distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  {algo_name}:")
    print(f"    Cluster Sizes: {cluster_sizes}")

def main():
    # Configuration
    input_path = "/vol/bitbucket/ahb24/tda_entailment_new/contradictions_only/contradiction_embeddings_SBERT_snli_10k_subset_balanced.pt"
    output_dir = Path("contradiction_analysis/clustering_results_with_data_extracts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Number of clusters to test
    k_values = [2, 3, 4, 5, 10]
    
    print("=" * 80)
    print("CONTRADICTION CLUSTERING ANALYSIS")
    print("=" * 80)
    
    # Load embeddings
    premise_embeddings, hypothesis_embeddings, metadata = load_embeddings(input_path)
    
    # Save raw contradiction data
    print("\n" + "=" * 80)
    print("EXTRACTING RAW DATA")
    print("=" * 80)
    save_raw_contradiction_data(metadata, premise_embeddings, hypothesis_embeddings, output_dir)
    
    # Create embedding spaces
    print("\n" + "=" * 80)
    print("CREATING EMBEDDING SPACES")
    print("=" * 80)
    embedding_spaces = create_embedding_spaces(premise_embeddings, hypothesis_embeddings)
    
    # Store all cluster assignments
    all_cluster_assignments = {}
    
    # Run clustering for each embedding space and k value
    print("\n" + "=" * 80)
    print("RUNNING CLUSTERING EXPERIMENTS")
    print("=" * 80)
    
    for space_name, embeddings in embedding_spaces.items():
        print(f"\n{space_name.upper()} EMBEDDING SPACE")
        print("-" * 80)
        
        all_cluster_assignments[space_name] = {}
        
        for k in k_values:
            print(f"\nk={k}")
            
            # Run clustering algorithms
            cluster_results = run_clustering(embeddings, k)
            all_cluster_assignments[space_name][k] = cluster_results
            
            # Print statistics and create visualizations
            for algo_name, labels in cluster_results.items():
                print_cluster_statistics(labels, algo_name, k)
                create_pacmap_visualization(embeddings, labels, algo_name, space_name, k, output_dir)
    
    # Save all cluster assignments
    save_cluster_assignments(all_cluster_assignments, metadata, output_dir)
    
    # Create summary report
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORT")
    print("=" * 80)
    
    summary = {
        'total_samples': len(premise_embeddings),
        'embedding_dimension': premise_embeddings.shape[1],
        'k_values_tested': k_values,
        'embedding_spaces': list(embedding_spaces.keys()),
        'clustering_algorithms': ['spectral', 'gmm', 'hierarchical'],
        'output_directory': str(output_dir)
    }
    
    summary_path = output_dir / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary report saved to: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTotal contradiction samples analyzed: {len(premise_embeddings)}")
    print(f"\nOutput files generated:")
    print(f"  1. contradiction_texts.csv - All contradiction text pairs")
    print(f"  2. contradiction_embeddings.npz - All embeddings")
    print(f"  3. contradiction_sample_100.csv - Sample for quick review")
    print(f"  4. cluster_assignments.csv - Cluster labels for all algorithms")
    print(f"  5. {len(k_values) * len(embedding_spaces) * 3} PaCMAP visualizations")
    print(f"  6. analysis_summary.json - Metadata about the analysis")
    print(f"\nAll files saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()