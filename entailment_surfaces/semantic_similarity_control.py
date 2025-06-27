"""
Step 1.2: Semantic Similarity Control Experiment
Critical validation: Does topology capture entailment structure vs just semantic similarity?

This experiment tests if PH-Dim differences are due to entailment relationships 
or merely semantic similarity patterns.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phd_method.src_phd.topology import ph_dim_from_distance_matrix, fast_ripser, calculate_ph_dim


class SemanticSimilarityController:
    """Controls for semantic similarity when analyzing entailment topology"""
    
    def __init__(self, data_path, embedding_model='all-MiniLM-L6-v2'):
        self.data_path = data_path
        self.similarity_model = SentenceTransformer(embedding_model)
        self.snli_data = None
        self.controlled_datasets = {}


    def load_snli_data(self):
        """Load SNLI dataset with BERT embeddings"""
        print("Loading SNLI data...")
        # Load your existing BERT embeddings
        data = torch.load(self.data_path)
        self.snli_data = data
        print(f"Loaded {len(data['labels'])} samples")

    def compute_semantic_similarities(self):
        """Compute semantic similarity scores for all premise-hypothesis pairs"""
        print("Computing semantic similarities...")
        
        premises = self.snli_data['premises']
        hypotheses = self.snli_data['hypotheses']
        
        # Get sentence embeddings for similarity computation
        premise_embeddings = self.similarity_model.encode(premises)
        hypothesis_embeddings = self.similarity_model.encode(hypotheses)
        
        # Compute pairwise cosine similarities
        similarities = np.array([
            cosine_similarity([p_emb], [h_emb])[0][0] 
            for p_emb, h_emb in zip(premise_embeddings, hypothesis_embeddings)
        ])
        
        self.snli_data['semantic_similarities'] = similarities
        print(f"Similarity range: {similarities.min():.3f} to {similarities.max():.3f}")

    
    def create_controlled_subsets(self, low_threshold=0.3, high_threshold=0.7):
        """
        Create controlled datasets with fixed semantic similarity thresholds
        
        Creates 9 groups across the full similarity spectrum:
        - Low similarity (< low_threshold) + {entailment, neutral, contradiction}
        - Mid similarity (low_threshold ≤ sim < high_threshold) + {entailment, neutral, contradiction}
        - High similarity (≥ high_threshold) + {entailment, neutral, contradiction}
        
        Args:
            low_threshold: Threshold below which pairs are considered "low similarity"
            high_threshold: Threshold above which pairs are considered "high similarity"
        """
        print(f"Creating controlled subsets: low<{low_threshold}, mid=[{low_threshold},{high_threshold}), high≥{high_threshold}...")
        
        labels = np.array(self.snli_data['labels'])
        similarities = self.snli_data['semantic_similarities']
        
        # Define similarity masks
        low_sim_mask = similarities < low_threshold
        mid_sim_mask = (similarities >= low_threshold) & (similarities < high_threshold)
        high_sim_mask = similarities >= high_threshold
        
        print(f"Low similarity pairs: {np.sum(low_sim_mask)} ({np.sum(low_sim_mask)/len(similarities)*100:.1f}%)")
        print(f"Mid similarity pairs: {np.sum(mid_sim_mask)} ({np.sum(mid_sim_mask)/len(similarities)*100:.1f}%)")
        print(f"High similarity pairs: {np.sum(high_sim_mask)} ({np.sum(high_sim_mask)/len(similarities)*100:.1f}%)")
        
        controlled_set = {
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'groups': {}
        }
        
        # Create 9 experimental groups
        groups = {
            'low_sim_entailment': (low_sim_mask) & (labels == 0),
            'low_sim_neutral': (low_sim_mask) & (labels == 1),
            'low_sim_contradiction': (low_sim_mask) & (labels == 2),
            'mid_sim_entailment': (mid_sim_mask) & (labels == 0),
            'mid_sim_neutral': (mid_sim_mask) & (labels == 1),
            'mid_sim_contradiction': (mid_sim_mask) & (labels == 2),
            'high_sim_entailment': (high_sim_mask) & (labels == 0),
            'high_sim_neutral': (high_sim_mask) & (labels == 1),
            'high_sim_contradiction': (high_sim_mask) & (labels == 2)
        }
        
        for group_name, mask in groups.items():
            if np.sum(mask) > 100:  # Ensure sufficient samples
                group_data = {
                    'indices': np.where(mask)[0],
                    'count': np.sum(mask),
                    'mean_similarity': similarities[mask].mean(),
                    'similarity_std': similarities[mask].std()
                }
                controlled_set['groups'][group_name] = group_data
                print(f"{group_name}: {group_data['count']} samples, "
                      f"sim={group_data['mean_similarity']:.3f}±{group_data['similarity_std']:.3f}")

            else:
                raise ValueError("not enough data samples")
        
        self.controlled_datasets = controlled_set



    """
    NEED TO ADJUST THIS FUNCTION DEPENDING ON WHAT THE RESULTS OF STEP 1.1 AND BEST DISTANCE METRICS AND
    EMBEDDING SPACES ARE!!!!

    def run_phd_analysis_on_controlled_sets(self, embedding_spaces=['bert_concat'], 
                                          distance_metrics=['euclidean', 'cosine']):
        """
        Run PH-Dim analysis on the 6 semantically controlled subsets
        
        Key Question: Do PH-Dim patterns persist when semantic similarity is controlled?
        """
        print("\n=== Running PH-Dim Analysis on Controlled Sets ===")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'thresholds': {
                'low_threshold': self.controlled_datasets['low_threshold'],
                'high_threshold': self.controlled_datasets['high_threshold']
            },
            'analysis_results': {}
        }
        
        for space_name in embedding_spaces:
            for metric in distance_metrics:
                print(f"\n--- Analyzing {space_name} + {metric} ---")
                
                space_results = {}
                
                # Analyze each controlled group (now 9 groups)
                for group_name, group_data in self.controlled_datasets['groups'].items():
                    if group_data['count'] < 200:  # Skip if insufficient samples
                        print(f"  {group_name}: Skipped (only {group_data['count']} samples)")
                        continue
                        
                    # Extract embeddings for this group
                    indices = group_data['indices']
                    group_embeddings = self.extract_embeddings(indices, space_name)
                    
                    # Compute PH-Dim for this controlled group
                    try:
                        phd_score = compute_persistent_homology_dimension(
                            group_embeddings,
                            distance_metric=metric,
                            min_points=min(200, len(group_embeddings)),
                            max_points=min(1000, len(group_embeddings))
                        )
                        
                        space_results[group_name] = {
                            'phd_score': phd_score,
                            'sample_count': group_data['count'],
                            'mean_similarity': group_data['mean_similarity'],
                            'similarity_std': group_data['similarity_std']
                        }
                        
                        print(f"  {group_name}: PH-Dim={phd_score:.4f} ({group_data['count']} samples)")
                        
                    except Exception as e:
                        print(f"  {group_name}: Failed - {e}")
                        space_results[group_name] = {'error': str(e)}
                
                results['analysis_results'][f"{space_name}_{metric}"] = space_results
        
        # Save results
        output_file = f"semantic_control_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return results
    

    AGAIN THIS FUNCTION BELOW NEEDS ADJUSTING TO BE ABLE TO EXTRACT CORRECT EMBEDDING SPACES (HOPEFULLY WE CAN JUST
    LOAD THEM FROM A FILE SAVED IN STEP 1.1)

    def extract_embeddings(self, indices, space_name):
        """Extract embeddings for specific indices and embedding space"""
        if space_name == 'bert_concat':
            # Concatenated [premise||hypothesis] embeddings
            premise_embs = self.snli_data['premise_embeddings'][indices]
            hypothesis_embs = self.snli_data['hypothesis_embeddings'][indices] 
            return torch.cat([premise_embs, hypothesis_embs], dim=-1).numpy()
        elif space_name == 'bert_difference':
            # Premise - hypothesis difference vectors
            premise_embs = self.snli_data['premise_embeddings'][indices]
            hypothesis_embs = self.snli_data['hypothesis_embeddings'][indices]
            return (premise_embs - hypothesis_embs).numpy()
        else:
            raise ValueError(f"Unknown embedding space: {space_name}")


    """

     def analyze_results(self, results):
        """
        Print raw results for manual analysis
        """
        print("\n=== SEMANTIC CONTROL RESULTS ===")
        
        for space_metric, groups in results['analysis_results'].items():
            print(f"\n{space_metric}:")
            print("-" * 50)
            
            # Extract and print all PH-Dim scores
            group_names = [
                'low_sim_entailment', 'low_sim_neutral', 'low_sim_contradiction',
                'mid_sim_entailment', 'mid_sim_neutral', 'mid_sim_contradiction', 
                'high_sim_entailment', 'high_sim_neutral', 'high_sim_contradiction'
            ]
            
            for group_name in group_names:
                if group_name in groups:
                    group_data = groups[group_name]
                    if 'phd_score' in group_data:
                        phd_score = group_data['phd_score']
                        sample_count = group_data['sample_count']
                        mean_sim = group_data['mean_similarity']
                        print(f"  {group_name:25}: PH-Dim={phd_score:.4f} (n={sample_count:4d}, sim={mean_sim:.3f})")
                    else:
                        print(f"  {group_name:25}: FAILED - {group_data.get('error', 'Unknown error')}")
                else:
                    print(f"  {group_name:25}: NO DATA")
            
            print()  # Empty line between metrics

def main():
    """Run the semantic similarity control experiment"""
    
    # Initialize controller
    controller = SemanticSimilarityController(
        data_path="data/processed/snli_full_standard_BERT.pt"
    )
    
    # Load data and compute similarities
    controller.load_snli_data()
    controller.compute_semantic_similarities()
    
    # Create controlled subsets with fixed thresholds
    controller.create_controlled_subsets(
        low_threshold=0.3,
        high_threshold=0.7
    )
    
    # Run PH-Dim analysis (use winning combinations from Step 1.1)
    results = controller.run_phd_analysis_on_controlled_sets(
        embedding_spaces=['bert_concat', 'bert_difference'],  # Update with winners from Step 1.1
        distance_metrics=['euclidean', 'cosine', 'manhattan']  # Update with winners from Step 1.1
    )
    
    # Print simple results for manual analysis
    controller.analyze_results(results)
    
    print("\n=== EXPERIMENT COMPLETE ===")






