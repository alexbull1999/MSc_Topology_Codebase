import numpy as np
import torch
import os
import json
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, pairwise_distances
from persim import PersistenceImager
from gph.python import ripser_parallel
from itertools import permutations

def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                            min_points=200,
                                            max_points=1000,
                                            point_jump=50,
                                            h_dim=0,
                                            alpha=1.0):
    """Compute PH-dimension and persistence diagrams from distance matrix"""
    assert dm.ndim == 2
    assert dm.shape[0] == dm.shape[1]

    test_n = range(min_points, max_points, point_jump)
    lengths = []
    all_diagrams = []

    for points_number in test_n:
        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]

        diagrams = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        all_diagrams.append(diagrams)

        d = diagrams[h_dim]
        d = d[d[:, 1] < np.inf]
        lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())

    lengths = np.array(lengths)

    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    
    ph_dimension = alpha / (1 - m)
    
    return ph_dimension, all_diagrams

def compute_distance_matrix(embeddings: torch.Tensor, metric: str) -> np.ndarray:
    """Compute distance matrix using specified metric"""
    embeddings_np = embeddings.detach().cpu().numpy()
    
    sklearn_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'correlation', 'braycurtis', 'canberra']
    
    if metric in sklearn_metrics:
        return pairwise_distances(embeddings_np, metric=metric)
    elif metric == 'minkowski_3':
        return pairwise_distances(embeddings_np, metric='minkowski', p=3)
    elif metric == 'minkowski_4':
        return pairwise_distances(embeddings_np, metric='minkowski', p=4)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def persistence_diagrams_to_images(all_diagrams):
    """Convert persistence diagrams to standardized images"""
    pimgr = PersistenceImager(
        pixel_size=0.5,
        birth_range=(0, 5),
        pers_range=(0, 5),
        kernel_params={'sigma': 0.3}
    )
    
    persistence_images = []
    
    for diagrams in all_diagrams:
        combined_image = np.zeros((20, 20))
        
        for dim in range(min(2, len(diagrams))):
            diagram = diagrams[dim]
            if len(diagram) > 0:
                finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
                if len(finite_diagram) > 0:
                    try:
                        img = pimgr.transform([finite_diagram])[0]
                        if img.shape != (20, 20):
                            from scipy.ndimage import zoom
                            zoom_factors = (20 / img.shape[0], 20 / img.shape[1])
                            img = zoom(img, zoom_factors)
                        combined_image += img
                    except:
                        continue
        
        if combined_image.max() > 0:
            combined_image = combined_image / combined_image.max()
        
        persistence_images.append(combined_image.flatten())
    
    return persistence_images

def calculate_clustering_accuracy(true_labels, predicted_labels):
    """Calculate best clustering accuracy over all label permutations"""
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    unique_predicted = np.unique(predicted_labels)
    best_accuracy = 0.0
    
    for perm in permutations(range(len(unique_predicted))):
        mapped_labels = np.zeros_like(predicted_labels)
        for i, cluster_id in enumerate(unique_predicted):
            mapped_labels[predicted_labels == cluster_id] = perm[i]
            
        accuracy = np.mean(true_labels == mapped_labels)
        best_accuracy = max(best_accuracy, accuracy)
        
    return best_accuracy

def evaluate_clustering_anli(round_name, split, seed=42, n_samples=10):
    """Evaluate point cloud clustering performance on ANLI dataset with multiple samples"""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/anli_processed/anli_{round_name}_{split}_SBERT.pt"
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing ANLI {round_name} {split}")
    print(f"{'='*60}")
    
    data = torch.load(data_path, weights_only=False)
    premise_embeddings = data['premise_embeddings']
    hypothesis_embeddings = data['hypothesis_embeddings']
    labels = data['labels']
    
    print(f"Total samples: {len(labels)}")
    
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")
    
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    entailment_mask = np.array([l == 'entailment' for l in labels])
    neutral_mask = np.array([l == 'neutral' for l in labels])
    contradiction_mask = np.array([l == 'contradiction' for l in labels])
    
    embedding_spaces = ['sbert_concat', 'lattice_containment']
    metrics = ['euclidean', 'cosine', 'braycurtis']
    
    results = {}
    
    for space_name in embedding_spaces:
        print(f"\nEmbedding Space: {space_name}")
        
        if space_name == 'sbert_concat':
            concatenated = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
            full_embeddings = {
                'entailment': concatenated[entailment_mask],
                'neutral': concatenated[neutral_mask],
                'contradiction': concatenated[contradiction_mask]
            }
        elif space_name == 'lattice_containment':
            epsilon = 1e-8
            containment = (premise_embeddings * hypothesis_embeddings) / (torch.abs(premise_embeddings) + torch.abs(hypothesis_embeddings) + epsilon)
            full_embeddings = {
                'entailment': containment[entailment_mask],
                'neutral': containment[neutral_mask],
                'contradiction': containment[contradiction_mask]
            }
        
        space_results = {}
        
        for metric in metrics:
            print(f"\n  Metric: {metric}")
            
            sample_results = []
            
            # Run n_samples clustering tests
            for run_idx in range(n_samples):
                np.random.seed(seed + run_idx)
                torch.manual_seed(seed + run_idx)
                
                all_persistence_images = []
                sample_labels = []
                
                # For each class, take 10 samples of 1000 points each
                for class_name in ['entailment', 'neutral', 'contradiction']:
                    class_idx = label_to_idx[class_name]
                    embeddings = full_embeddings[class_name]
                    
                    if len(embeddings) < 200:
                        continue
                    
                    # Take 10 samples per class
                    for sample_idx in range(10):
                        # Sample up to 1000 points
                        max_points = min(1000, len(embeddings))
                        indices = torch.randperm(len(embeddings))[:max_points]
                        sampled_embeddings = embeddings[indices]
                        
                        try:
                            # Compute distance matrix
                            distance_matrix = compute_distance_matrix(sampled_embeddings, metric)
                            
                            # Get PH-dim and ALL persistence diagrams
                            ph_dim, all_diagrams = ph_dim_and_diagrams_from_distance_matrix(
                                distance_matrix,
                                min_points=200,
                                max_points=len(sampled_embeddings),
                                point_jump=50,
                                h_dim=0,
                                alpha=1.0
                            )
                            
                            # Convert persistence diagrams to images (H0 and H1 combined)
                            if len(all_diagrams) > 0:
                                persistence_image = persistence_diagrams_to_images([all_diagrams[0]])
                                if len(persistence_image) > 0:
                                    all_persistence_images.append(persistence_image[0])
                                    sample_labels.append(class_idx)
                            
                        except Exception as e:
                            print(f"    Run {run_idx+1}, {class_name}, sample {sample_idx+1}: Error - {e}")
                
                # Cluster the persistence images (should have 30: 10 per class)
                if len(all_persistence_images) >= 3:
                    X = np.vstack(all_persistence_images)
                    
                    kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
                    predicted_labels = kmeans.fit_predict(X)
                    
                    accuracy = calculate_clustering_accuracy(sample_labels, predicted_labels)
                    
                    try:
                        if len(set(predicted_labels)) > 1 and len(X) > 3:
                            sil_score = silhouette_score(X, predicted_labels)
                        else:
                            sil_score = 0.0
                    except:
                        sil_score = 0.0
                    
                    ari = adjusted_rand_score(sample_labels, predicted_labels)
                    perfect_clustering = (accuracy == 1.0)
                    
                    sample_results.append({
                        'accuracy': float(accuracy),
                        'silhouette': float(sil_score),
                        'ari': float(ari),
                        'perfect_clustering': perfect_clustering
                    })
                    
                    print(f"    Run {run_idx+1}: Acc={accuracy:.3f}, Sil={sil_score:.3f}, Perfect={perfect_clustering}")
            
            # Aggregate results across all samples
            if sample_results:
                accuracies = [r['accuracy'] for r in sample_results]
                silhouettes = [r['silhouette'] for r in sample_results]
                perfect_clustering = [r['perfect_clustering'] for r in sample_results]
                
                space_results[metric] = {
                    'n_samples': len(sample_results),
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'silhouette_mean': float(np.mean(silhouettes)),
                    'silhouette_std': float(np.std(silhouettes)),
                    'perfect_clustering_rate': float(np.mean(perfect_clustering)),
                    'sample_results': sample_results
                }
                
                print(f"\n  Summary for {metric}:")
                print(f"    Accuracy: {space_results[metric]['accuracy_mean']:.3f} ± {space_results[metric]['accuracy_std']:.3f}")
                print(f"    Silhouette: {space_results[metric]['silhouette_mean']:.3f} ± {space_results[metric]['silhouette_std']:.3f}")
                print(f"    Perfect clustering rate: {space_results[metric]['perfect_clustering_rate']:.1%}")
            else:
                space_results[metric] = None
        
        results[space_name] = space_results
    
    return {
        'round': round_name,
        'split': split,
        'seed': seed,
        'n_samples_per_metric': n_samples,
        'total_samples': len(labels),
        'label_counts': label_counts,
        'results': results
    }

if __name__ == "__main__":
    SEED = 42
    N_SAMPLES = 10
    
    rounds = ['R1', 'R2', 'R3']
    splits = ['train']
    
    print(f"Running with seed: {SEED}")
    print(f"Number of samples per metric: {N_SAMPLES}")
    print("="*60)
    
    all_results = []
    
    for round_name in rounds:
        for split in splits:
            result = evaluate_clustering_anli(round_name, split, seed=SEED, n_samples=N_SAMPLES)
            if result:
                all_results.append(result)
    
    output_dir = "entailment_surfaces/anli_results/clustering"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/anli_clustering_validation_train_seed{SEED}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")