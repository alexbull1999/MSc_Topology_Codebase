# data/extract_xnli_combined.py
"""
Extract XNLI dataset combining validation and test splits into single files
This gives ~7,500 samples per language for clustering experiments
"""

from datasets import load_dataset
import json
import os
from pathlib import Path
from collections import defaultdict

# Set cache directory to unlimited storage location
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'

def extract_xnli_combined(language='en'):
    """
    Extract XNLI dataset combining validation and test splits
    
    Args:
        language: Language code (e.g., 'en', 'zh', 'ar', 'de', etc.)
    
    Output:
        Single JSON file per language with ~7,500 samples (2,490 val + 5,010 test)
        Format: [[premise, hypothesis, label], ...]
        where label is 'entailment', 'neutral', or 'contradiction'
    """
    
    print(f"Loading XNLI dataset for language: {language}")
    
    # Set cache directory
    cache_dir = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
    
    # Load specific language
    dataset = load_dataset("xnli", language, cache_dir=cache_dir)
    
    # Label mapping (XNLI uses integer labels)
    label_map = {
        0: 'entailment',
        1: 'neutral', 
        2: 'contradiction'
    }
    
    # Combine validation and test data
    combined_data = []
    
    for split_name in ['validation', 'test']:
        print(f"Processing {split_name} split...")
        
        for item in dataset[split_name]:
            premise = item['premise']
            hypothesis = item['hypothesis']
            label = label_map[item['label']]
            
            combined_data.append([premise, hypothesis, label])
    
    # Create output directory
    output_dir = Path(f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/raw/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined data
    output_path = output_dir / f"xnli_{language}_combined.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(combined_data)} samples to {output_path}")
    
    # Print label distribution
    label_counts = defaultdict(int)
    for item in combined_data:
        label_counts[item[2]] += 1
    
    print(f"Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(combined_data)*100:.1f}%)")
    
    return len(combined_data)

def main():
    """Extract XNLI for all 15 languages"""
    
    # All XNLI languages
    languages = [
        'en',  # English
        'ar',  # Arabic
        'bg',  # Bulgarian
        'de',  # German
        'el',  # Greek
        'es',  # Spanish
        'fr',  # French
        'hi',  # Hindi
        'ru',  # Russian
        'sw',  # Swahili
        'th',  # Thai
        'tr',  # Turkish
        'ur',  # Urdu
        'vi',  # Vietnamese
        'zh',  # Chinese
    ]
    
    total_samples = {}
    
    for lang in languages:
        print("=" * 60)
        print(f"Extracting XNLI - {lang}")
        print("=" * 60)
        total_samples[lang] = extract_xnli_combined(language=lang)
        print()
    
    # Summary
    print("=" * 60)
    print("EXTRACTION COMPLETE - SUMMARY")
    print("=" * 60)
    for lang, count in total_samples.items():
        print(f"{lang}: {count} samples")
    print(f"\nTotal across all languages: {sum(total_samples.values())} samples")
    print(f"Average per language: {sum(total_samples.values())/len(total_samples):.0f} samples")

if __name__ == "__main__":
    main()