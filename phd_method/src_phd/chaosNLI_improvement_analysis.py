#!/usr/bin/env python3
"""
ChaosNLI Improvement Analysis - FIXED VERSION

This script identifies the best examples where our topological method 
shows the biggest improvements over baseline models on ChaosNLI datasets.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

class ChaosNLIImprovementAnalyzer:
    """Find examples where topological fusion shows biggest improvements"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        self.model = None
        
        # Load the trained CNN
        self._load_model()
        
    def _load_model(self):
        """Load the trained persistence CNN"""
        from ALL_TRAIN_CHUNKS_chaosNLI_hybrid_SOTA_PersimCNN_image_classification_SNLI_MNLI_separate_eval import PersistenceImageCNN
        
        print(f"Loading trained model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model = PersistenceImageCNN()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded - Best val acc: {checkpoint['best_val_acc']:.3f}")
    
    def load_chaosnli_data(self):
        """Load ChaosNLI datasets with all required information"""
        
        print("Loading ChaosNLI datasets...")
        
        # Load ChaosNLI-S (SNLI) persistence data
        snli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_persistence_images.pkl"
        with open(snli_persistence_path, 'rb') as f:
            snli_data = pickle.load(f)
        
        # Load ChaosNLI-M (MNLI) persistence data
        mnli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/MNLI_ORDER_ASYMM_MODELS_precomputed_chaosnli_mnli_matched_persistence_images.pkl"
        with open(mnli_persistence_path, 'rb') as f:
            mnli_data = pickle.load(f)
        
        # Load text data from original Arrow files
        text_data_dict = self._load_text_from_arrow_files()
        
        datasets = {
            'snli': {
                'persistence_images': snli_data['persistence_images'],
                'uids': snli_data['uids'], 
                'label_distributions': snli_data['label_distributions'],
                'majority_labels': snli_data['majority_labels'],
                'entropies': snli_data['entropies'],
                'texts': self._match_texts_with_persistence_uids(snli_data['uids'], text_data_dict, 'snli')
            },
            'mnli': {
                'persistence_images': mnli_data['persistence_images'],
                'uids': mnli_data['uids'],
                'label_distributions': mnli_data['label_distributions'], 
                'majority_labels': mnli_data['majority_labels'],
                'entropies': mnli_data['entropies'],
                'texts': self._match_texts_with_persistence_uids(mnli_data['uids'], text_data_dict, 'mnli')
            }
        }
        
        print(f"Loaded ChaosNLI-S: {len(datasets['snli']['uids'])} samples")
        print(f"Loaded ChaosNLI-M: {len(datasets['mnli']['uids'])} samples")
        
        return datasets
    
    def _load_text_from_arrow_files(self):
        """Load text data from the original Arrow files using the datasets library"""
        
        print("Loading text from original Arrow files using datasets library...")
        
        text_data = {'snli': {}, 'mnli': {}}
        
        # SNLI: Load from HuggingFace datasets to get original pairIDs
        print("Loading SNLI from HuggingFace datasets...")
        try:
            from datasets import load_dataset
            snli_dataset = load_dataset('snli', split='test')
            
            print(f"SNLI dataset loaded. Checking available fields...")
            sample_item = snli_dataset[0]
            print(f"Available fields: {list(sample_item.keys())}")
            
            for i, item in enumerate(snli_dataset):
                if item['label'] != -1:  # Filter out invalid labels
                    # Check what UID field exists
                    if 'pairID' in item:
                        uid = item['pairID']
                    elif 'pair_id' in item:
                        uid = item['pair_id']
                    else:
                        # Construct UID from index - this might match your format
                        uid = f"{i}.jpg#0r1e"
                    
                    text_data['snli'][uid] = {
                        'premise': item['premise'],
                        'hypothesis': item['hypothesis'],
                        'gold_label': item['label']
                    }
            
            print(f"Loaded {len(text_data['snli'])} SNLI examples")
            print(f"Sample SNLI UIDs: {list(text_data['snli'].keys())[:3]}")
            
        except Exception as e:
            print(f"Error loading SNLI from HuggingFace: {e}")
            print("Falling back to ChaosNLI text file...")
        
        try:
            from datasets import Dataset
                    
            # Try MNLI if path exists
            mnli_arrow_path = "MSc_Topology_Codebase/data/raw/mnli/validation_matched/data-00000-of-00001.arrow"
            if Path(mnli_arrow_path).exists():
                try:
                    dataset = Dataset.from_file(mnli_arrow_path)
                    df = dataset.to_pandas()
                    
                    print(f"Loaded MNLI Arrow file: {len(df)} rows")
                    
                    for idx, row in df.iterrows():
                        if 'pairID' in df.columns:
                            uid = row['pairID']
                        else:
                            uid = f"mnli_{idx}"
                        
                        text_data['mnli'][uid] = {
                            'premise': row.get('premise', ''),
                            'hypothesis': row.get('hypothesis', ''),
                            'gold_label': row.get('label', '')
                        }
                    
                    print(f"Created MNLI text mappings for {len(text_data['mnli'])} examples")
                    
                except Exception as e:
                    print(f"Error loading MNLI Arrow file: {e}")
            
            # If we didn't get good data, try the JSON files you created
            if not text_data['snli'] or not text_data['mnli']:
                print("Arrow files didn't work, trying JSON extraction files...")
                return self._load_text_from_json_extracts()
                    
        except ImportError:
            print("Datasets library not available, trying JSON extraction files...")
            return self._load_text_from_json_extracts()
        
        return text_data
    
    
    def _load_text_fallback(self):
        """Fallback method using ChaosNLI text files"""
        
        print("Using ChaosNLI text files as fallback...")
        
        text_data = {'snli': {}, 'mnli': {}}
        
        # Load ChaosNLI SNLI text
        snli_text_file = "ChaosNLI/data/chaosNLI_snli.jsonl"
        if Path(snli_text_file).exists():
            with open(snli_text_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    uid = entry['uid']
                    text_data['snli'][uid] = {
                        'premise': entry['example']['premise'],
                        'hypothesis': entry['example']['hypothesis'],
                        'gold_label': entry['old_label']
                    }
            print(f"Loaded {len(text_data['snli'])} SNLI texts from ChaosNLI")
        
        # Load ChaosNLI MNLI text
        mnli_text_file = "ChaosNLI/data/chaosNLI_mnli_m.jsonl"
        if Path(mnli_text_file).exists():
            with open(mnli_text_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    uid = entry['uid']
                    text_data['mnli'][uid] = {
                        'premise': entry['example']['premise'],
                        'hypothesis': entry['example']['hypothesis'],
                        'gold_label': entry['old_label']
                    }
            print(f"Loaded {len(text_data['mnli'])} MNLI texts from ChaosNLI")
        
        return text_data
    
    def _match_texts_with_persistence_uids(self, persistence_uids: List[str], text_data_dict: Dict, dataset_name: str) -> Dict:
        """Match persistence UIDs with text data"""
        
        text_data = text_data_dict.get(dataset_name.lower(), {})
        matched_texts = {}
        found_count = 0
        
        print(f"Matching {len(persistence_uids)} {dataset_name} UIDs with text data")
        print(f"Available text UIDs: {len(text_data)}")
        print(f"Sample persistence UIDs: {persistence_uids[:3]}")
        print(f"Sample text UIDs: {list(text_data.keys())[:3]}")
        
        for uid in persistence_uids:
            if uid in text_data:
                matched_texts[uid] = text_data[uid]
                found_count += 1
            else:
                # Create placeholder
                matched_texts[uid] = {
                    'premise': f"[Text not found for {uid}]",
                    'hypothesis': f"[Text not found for {uid}]",
                    'gold_label': 'unknown'
                }
        
        match_rate = found_count / len(persistence_uids) * 100
        print(f"Matched {found_count}/{len(persistence_uids)} ({match_rate:.1f}%)")
        
        return matched_texts
    
    def load_published_predictions(self) -> Dict:
        """Load published baseline model predictions"""
        
        predictions_file = "ChaosNLI/data/model_predictions/model_predictions_for_snli_mnli.json"
        
        print(f"Loading published predictions from {predictions_file}")
        
        if not Path(predictions_file).exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        print(f"Available baseline models: {list(predictions.keys())}")
        return predictions
    
    def get_topological_predictions(self, persistence_images: np.ndarray) -> Dict:
        """Get predictions from our topological CNN"""
        
        print(f"Getting topological predictions for {len(persistence_images)} samples")
        
        self.model.eval()
        predictions = []
        probabilities = []
        logits = []
        
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(persistence_images), batch_size):
                batch_images = persistence_images[i:i+batch_size]
                
                # Reshape to 30x30 for CNN
                batch_tensor = torch.FloatTensor(batch_images).reshape(-1, 1, 30, 30).to(self.device)
                
                batch_logits = self.model(batch_tensor)
                batch_probs = F.softmax(batch_logits, dim=1)
                batch_preds = torch.argmax(batch_logits, dim=1)
                
                logits.extend(batch_logits.cpu().numpy())
                probabilities.extend(batch_probs.cpu().numpy())
                predictions.extend(batch_preds.cpu().numpy())
        
        return {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'logits': np.array(logits)
        }
    
    def find_fusion_weights(self, baseline_logits: np.ndarray, 
                           topological_logits: np.ndarray,
                           human_distributions: np.ndarray,
                           weight_range: np.ndarray = np.linspace(0.0, 1.0, 11)) -> Tuple[float, np.ndarray]:
        """Find optimal fusion weights for this dataset"""
        
        best_alpha = 0.0
        best_jsd = float('inf')
        best_fused_probs = None
        
        for alpha in weight_range:
            # Weighted fusion: (1-α) * baseline + α * topological
            fused_logits = (1 - alpha) * baseline_logits + alpha * topological_logits
            fused_probs = F.softmax(torch.FloatTensor(fused_logits), dim=1).numpy()
            
            # Calculate average JSD
            jsd_scores = []
            for i in range(len(fused_probs)):
                pred_dist = np.clip(fused_probs[i] + 1e-10, 1e-10, 1.0)
                human_dist = np.clip(human_distributions[i] + 1e-10, 1e-10, 1.0)
                pred_dist = pred_dist / pred_dist.sum()
                human_dist = human_dist / human_dist.sum()
                
                jsd = jensenshannon(pred_dist, human_dist)
                jsd_scores.append(jsd)
            
            avg_jsd = np.mean(jsd_scores)
            
            if avg_jsd < best_jsd:
                best_jsd = avg_jsd
                best_alpha = alpha
                best_fused_probs = fused_probs
        
        return best_alpha, best_fused_probs
    
    def calculate_improvements(self, baseline_logits: np.ndarray,
                             fused_probabilities: np.ndarray, 
                             human_distributions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate per-sample improvements (JSD and KL divergence reduction)"""
        
        baseline_probs = F.softmax(torch.FloatTensor(baseline_logits), dim=1).numpy()
        
        jsd_improvements = []
        kl_improvements = []
        
        for i in range(len(baseline_probs)):
            # Normalize distributions
            baseline_dist = np.clip(baseline_probs[i] + 1e-10, 1e-10, 1.0)
            fused_dist = np.clip(fused_probabilities[i] + 1e-10, 1e-10, 1.0)  
            human_dist = np.clip(human_distributions[i] + 1e-10, 1e-10, 1.0)
            
            baseline_dist = baseline_dist / baseline_dist.sum()
            fused_dist = fused_dist / fused_dist.sum()
            human_dist = human_dist / human_dist.sum()
            
            # Calculate JSD improvements
            baseline_jsd = jensenshannon(baseline_dist, human_dist)
            fused_jsd = jensenshannon(fused_dist, human_dist)
            jsd_improvement = baseline_jsd - fused_jsd  # Positive = improvement
            
            # Calculate KL divergence improvements  
            baseline_kl = entropy(human_dist, baseline_dist)
            fused_kl = entropy(human_dist, fused_dist)
            kl_improvement = baseline_kl - fused_kl  # Positive = improvement
            
            jsd_improvements.append(jsd_improvement)
            kl_improvements.append(kl_improvement)
        
        return np.array(jsd_improvements), np.array(kl_improvements)
    
    def find_best_examples_per_class(self, dataset_name: str, dataset: Dict, 
                                   baseline_model: str, n_examples: int = 5) -> List[Dict]:
        """Find best improvement examples for each class in a dataset"""
        
        print(f"\nFinding best examples for {dataset_name} with {baseline_model}")
        
        # Get baseline predictions for this model
        published_predictions = self.load_published_predictions()
        baseline_preds = published_predictions[baseline_model]
        
        # Match data with baseline predictions
        matched_baseline_logits = []
        matched_indices = []
        
        for i, uid in enumerate(dataset['uids']):
            if uid in baseline_preds:
                matched_baseline_logits.append(baseline_preds[uid]['logits'])
                matched_indices.append(i)
        
        if not matched_baseline_logits:
            print(f"No matches found for {baseline_model} on {dataset_name}")
            return []
        
        matched_baseline_logits = np.array(matched_baseline_logits)
        matched_indices = np.array(matched_indices)
        
        # Get corresponding data
        matched_persistence = dataset['persistence_images'][matched_indices]
        matched_human_dists = dataset['label_distributions'][matched_indices]
        matched_uids = [dataset['uids'][i] for i in matched_indices]
        
        # Get topological predictions
        topo_results = self.get_topological_predictions(matched_persistence)
        
        # Find best fusion weights
        best_alpha, fused_probs = self.find_fusion_weights(
            matched_baseline_logits, topo_results['logits'], matched_human_dists
        )
        
        print(f"Best alpha for {dataset_name}: {best_alpha:.2f}")
        
        # Calculate improvements per sample
        jsd_improvements, kl_improvements = self.calculate_improvements(
            matched_baseline_logits, fused_probs, matched_human_dists
        )
        
        # Group by majority class and find top examples for each
        best_examples = []
        
        for class_idx in range(3):
            class_name = ['entailment', 'neutral', 'contradiction'][class_idx]
            
            # Find samples where this is the majority class
            class_mask = np.argmax(matched_human_dists, axis=1) == class_idx
            
            if not np.any(class_mask):
                print(f"No {class_name} examples found")
                continue
            
            class_jsd_improvements = jsd_improvements[class_mask]
            class_kl_improvements = kl_improvements[class_mask]
            class_matched_indices = matched_indices[class_mask]
            class_uids = [matched_uids[i] for i in np.where(class_mask)[0]]
            
            # Get top N examples for this class (ranked by JSD improvement)
            top_indices = np.argsort(class_jsd_improvements)[-n_examples:][::-1]
            
            for rank, idx in enumerate(top_indices):
                original_idx = class_matched_indices[idx]
                uid = class_uids[idx]
                jsd_improvement = class_jsd_improvements[idx]
                kl_improvement = class_kl_improvements[idx]
                
                # Get all relevant information
                example = {
                    'dataset': dataset_name,
                    'class': class_name,
                    'rank': rank + 1,
                    'uid': uid,
                    'jsd_improvement': jsd_improvement,
                    'kl_improvement': kl_improvement,
                    'best_alpha': best_alpha,
                    
                    # Text data
                    'premise': dataset['texts'][uid]['premise'],
                    'hypothesis': dataset['texts'][uid]['hypothesis'],
                    
                    # Label distributions
                    'human_distribution': matched_human_dists[np.where(class_mask)[0][idx]],
                    'human_entropy': dataset['entropies'][original_idx],
                    
                    # Predictions
                    'baseline_logits': matched_baseline_logits[np.where(class_mask)[0][idx]],
                    'baseline_probs': F.softmax(torch.FloatTensor(matched_baseline_logits[np.where(class_mask)[0][idx]]), dim=0).numpy(),
                    'topological_logits': topo_results['logits'][np.where(class_mask)[0][idx]], 
                    'topological_probs': topo_results['probabilities'][np.where(class_mask)[0][idx]],
                    'fused_probs': fused_probs[np.where(class_mask)[0][idx]],
                    
                    # Metrics
                    'baseline_jsd': jensenshannon(
                        F.softmax(torch.FloatTensor(matched_baseline_logits[np.where(class_mask)[0][idx]]), dim=0).numpy(),
                        matched_human_dists[np.where(class_mask)[0][idx]]
                    ),
                    'fused_jsd': jensenshannon(
                        fused_probs[np.where(class_mask)[0][idx]],
                        matched_human_dists[np.where(class_mask)[0][idx]]
                    ),
                    'baseline_kl': entropy(
                        matched_human_dists[np.where(class_mask)[0][idx]],
                        F.softmax(torch.FloatTensor(matched_baseline_logits[np.where(class_mask)[0][idx]]), dim=0).numpy()
                    ),
                    'fused_kl': entropy(
                        matched_human_dists[np.where(class_mask)[0][idx]],
                        fused_probs[np.where(class_mask)[0][idx]]
                    )
                }
                
                best_examples.append(example)
                
            print(f"Found {len(top_indices)} {class_name} examples")
        
        return best_examples
    
    def run_analysis(self, baseline_models: List[str] = ['bert-base', 'roberta-base'], 
                    n_examples: int = 5) -> Dict:
        """Run complete analysis to find best improvement examples"""
        
        print("="*80)
        print("CHAOSNLI IMPROVEMENT ANALYSIS")  
        print("="*80)
        
        # Load datasets
        datasets = self.load_chaosnli_data()
        
        all_results = {}
        
        for model_name in baseline_models:
            print(f"\n{'='*60}")
            print(f"ANALYZING IMPROVEMENTS OVER: {model_name.upper()}")
            print(f"{'='*60}")
            
            model_results = {}
            
            for dataset_name in ['snli', 'mnli']:
                examples = self.find_best_examples_per_class(
                    dataset_name, datasets[dataset_name], model_name, n_examples
                )
                model_results[dataset_name] = examples
            
            all_results[model_name] = model_results
        
        return all_results
    
    def print_example_summary(self, results: Dict):
        """Print a nice summary of the best examples found"""
        
        print(f"\n{'='*100}")
        print("BEST IMPROVEMENT EXAMPLES SUMMARY")
        print(f"{'='*100}")
        
        for model_name, model_results in results.items():
            print(f"\n{model_name.upper()} IMPROVEMENTS:")
            print("-" * 50)
            
            for dataset_name, examples in model_results.items():
                print(f"\n{dataset_name.upper()}:")
                
                for example in examples:
                    print(f"  {example['class'].title()} #{example['rank']}:")
                    print(f"    JSD Improvement: {example['jsd_improvement']:.4f}")
                    print(f"    KL Improvement: {example['kl_improvement']:.4f}")
                    print(f"    Alpha: {example['best_alpha']:.2f}")
                    print(f"    Premise: {example['premise'][:80]}...")
                    print(f"    Hypothesis: {example['hypothesis'][:80]}...")
                    print(f"    JSD: {example['baseline_jsd']:.4f} → {example['fused_jsd']:.4f}")
                    print(f"    KL: {example['baseline_kl']:.4f} → {example['fused_kl']:.4f}")
                    print()
    
    def save_results(self, results: Dict, output_path: str = "MSc_Topology_Codebase/phd_method/chaosNLI_individual_analysis/chaosnli_improvement_analysis.json"):
        """Save analysis results to file"""
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        
        for model_name, model_results in results.items():
            json_results[model_name] = {}
            
            for dataset_name, examples in model_results.items():
                json_examples = []
                
                for example in examples:
                    json_example = example.copy()
                    
                    # Convert numpy arrays to lists
                    for key in ['human_distribution', 'baseline_logits', 'baseline_probs', 
                              'topological_logits', 'topological_probs', 'fused_probs']:
                        if key in json_example:
                            json_example[key] = json_example[key].tolist()
                    
                    json_examples.append(json_example)
                
                json_results[model_name][dataset_name] = json_examples
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """Run the improvement analysis"""
    
    # You'll need to update this path to your trained model
    model_path = "MSc_Topology_Codebase/phd_method/chaosNLI_models/chaosnli_persistence_cnn.pt"
    
    analyzer = ChaosNLIImprovementAnalyzer(model_path)
    
    print("Starting ChaosNLI improvement analysis with Arrow file text loading...")
    
    # Run analysis
    results = analyzer.run_analysis(
        baseline_models=['bert-base', 'roberta-base'],
        n_examples=3  # Start with fewer examples for testing
    )
    
    # Print summary
    analyzer.print_example_summary(results)
    
    # Save results
    analyzer.save_results(results)
    
    print("\nImprovement analysis completed!")
    print("Check the match rates in the output to see if text loading worked correctly.")


if __name__ == "__main__":
    main()