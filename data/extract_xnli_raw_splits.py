# data/extract_xnli_separate_splits.py
"""
Extract XNLI data with separate validation and test splits
For unsupervised clustering experiments where we optimize on validation, evaluate on test
"""

import os

# CRITICAL: Set cache directories BEFORE importing anything from HuggingFace
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'

from datasets import load_dataset
import json
from pathlib import Path



def extract_xnli_separate_splits():
    """Extract XNLI with separate validation and test splits for each language"""
    
    # All XNLI languages
    languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    
    # Label mapping
    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    
    output_base = Path("/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/raw_splits/")
    output_base.mkdir(parents=True, exist_ok=True)
    
    for lang in languages:
        print(f"\nProcessing {lang}...")
        
        try:
            # Load XNLI for this language
            dataset = load_dataset("xnli", lang)
            
            # Process validation split
            validation_data = []
            for example in dataset['validation']:
                premise = example['premise']
                hypothesis = example['hypothesis']
                label = label_map[example['label']]
                validation_data.append([premise, hypothesis, label])
            
            # Process test split
            test_data = []
            for example in dataset['test']:
                premise = example['premise']
                hypothesis = example['hypothesis']
                label = label_map[example['label']]
                test_data.append([premise, hypothesis, label])
            
            # Save validation split
            val_output_path = output_base / f"xnli_{lang}_validation.json"
            with open(val_output_path, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, ensure_ascii=False, indent=2)
            print(f"  Saved validation: {len(validation_data)} samples -> {val_output_path}")
            
            # Save test split
            test_output_path = output_base / f"xnli_{lang}_test.json"
            with open(test_output_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"  Saved test: {len(test_data)} samples -> {test_output_path}")
            
            # Print label distribution
            val_labels = [item[2] for item in validation_data]
            test_labels = [item[2] for item in test_data]
            
            val_counts = {label: val_labels.count(label) for label in ['entailment', 'neutral', 'contradiction']}
            test_counts = {label: test_labels.count(label) for label in ['entailment', 'neutral', 'contradiction']}
            
            print(f"  Validation distribution: {val_counts}")
            print(f"  Test distribution: {test_counts}")
            
        except Exception as e:
            print(f"  Error processing {lang}: {e}")
            continue
    
    print("\n" + "="*60)
    print("XNLI extraction with separate splits complete!")
    print("="*60)

if __name__ == "__main__":
    extract_xnli_separate_splits()