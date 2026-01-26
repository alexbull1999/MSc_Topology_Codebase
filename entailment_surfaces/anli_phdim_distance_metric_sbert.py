import torch
import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phd_method.src_phd.topology import fast_ripser

def compute_phdim_anli(round_name, split, seed=42):
    """Compute PH-Dim for ANLI dataset"""
    
    # Set all random seeds
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
    
    n_samples = len(labels)
    print(f"Total samples: {n_samples}")
    
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")
    
    concatenated = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
    
    entailment_mask = np.array([l == 'entailment' for l in labels])
    neutral_mask = np.array([l == 'neutral' for l in labels])
    contradiction_mask = np.array([l == 'contradiction' for l in labels])
    
    metrics = ['euclidean', 'cosine', 'braycurtis']
    embedding_spaces = ['sbert_concat', 'lattice_containment']
    
    print(f"\n{'='*60}")
    print("PH-Dim Analysis")
    print(f"{'='*60}")
    
    phdim_results = {}
    
    for space_name in embedding_spaces:
        print(f"\nEmbedding Space: {space_name}")
        
        if space_name == 'sbert_concat':
            space_embeddings = {
                'entailment': concatenated[entailment_mask],
                'neutral': concatenated[neutral_mask],
                'contradiction': concatenated[contradiction_mask]
            }
        elif space_name == 'lattice_containment':
            epsilon = 1e-8
            containment = (premise_embeddings * hypothesis_embeddings) / (torch.abs(premise_embeddings) + torch.abs(hypothesis_embeddings) + epsilon)
            space_embeddings = {
                'entailment': containment[entailment_mask],
                'neutral': containment[neutral_mask],
                'contradiction': containment[contradiction_mask]
            }
        
        space_results = {}
        
        for metric in metrics:
            print(f"\n  Metric: {metric}")
            
            metric_results = {}
            
            for class_name, embeddings in space_embeddings.items():
                if len(embeddings) < 200:
                    print(f"    {class_name}: Not enough samples ({len(embeddings)} < 200)")
                    metric_results[class_name] = None
                    continue
                
                # Store original length before sampling
                original_len = len(embeddings)
                max_points = min(1000, original_len)
                
                # Sample if needed
                if original_len > max_points:
                    torch.manual_seed(seed)
                    indices = torch.randperm(original_len)[:max_points]
                    embeddings = embeddings[indices]
                
                embeddings_np = embeddings.detach().cpu().numpy()
                
                try:
                    phd = fast_ripser(
                        embeddings_np,
                        min_points=200,
                        max_points=min(1000, original_len),  # Use original length, not sampled
                        point_jump=50,
                        h_dim=0,
                        alpha=1.0,
                        seed=seed,
                        metric=metric
                    )
                    
                    print(f"    {class_name} PH-Dim: {phd:.4f} (computed on {len(embeddings_np)} points, max_points={min(1000, original_len)})")
                    metric_results[class_name] = float(phd)
                    
                except Exception as e:
                    print(f"    {class_name}: Error computing PH-Dim: {e}")
                    metric_results[class_name] = None
            
            space_results[metric] = metric_results
        
        phdim_results[space_name] = space_results
    
    return {
        'round': round_name,
        'split': split,
        'seed': seed,
        'n_samples': n_samples,
        'label_counts': label_counts,
        'phdim_results': phdim_results
    }

if __name__ == "__main__":
    # Change this seed value to test reproducibility
    SEED = 555
    
    rounds = ['R1', 'R2', 'R3']
    splits = ['train']
    
    print(f"Running with seed: {SEED}")
    print("="*60)
    
    all_results = []
    
    for round_name in rounds:
        for split in splits:
            result = compute_phdim_anli(round_name, split, seed=SEED)
            if result:
                all_results.append(result)
    
    output_dir = "entailment_surfaces/anli_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/anli_phdim_train_seed{SEED}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")