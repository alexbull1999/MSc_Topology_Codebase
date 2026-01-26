from datasets import load_dataset
import json
import os
from pathlib import Path

# Set cache directory to unlimited storage location
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'

def download_anli():
    """Download ANLI dataset from HuggingFace"""
    
    # Create cache directory
    cache_dir = Path('/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directory
    base_dir = Path("/vol/bitbucket/ahb24/tda_entailment_new/anli_raw")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download ANLI dataset from HuggingFace (only need to do this once)
    print(f"\nDownloading ANLI dataset...")
    dataset = load_dataset(
        "anli",
        cache_dir=str(cache_dir)
    )
    
    print(f"Available splits: {list(dataset.keys())}")
    
    # Map HuggingFace splits to round directories
    split_mapping = {
        'train_r1': ('R1', 'train'),
        'train_r2': ('R2', 'train'),
        'train_r3': ('R3', 'train'),
        'dev_r1': ('R1', 'dev'),
        'dev_r2': ('R2', 'dev'),
        'dev_r3': ('R3', 'dev'),
        'test_r1': ('R1', 'test'),
        'test_r2': ('R2', 'test'),
        'test_r3': ('R3', 'test')
    }
    
    for hf_split, (round_name, output_split) in split_mapping.items():
        if hf_split not in dataset:
            print(f"Split {hf_split} not found in dataset")
            continue
        
        # Create round directory
        round_dir = base_dir / round_name
        round_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = round_dir / f"{output_split}.jsonl"
        
        if output_file.exists():
            print(f"File already exists: {output_file}")
            continue
        
        print(f"Saving {round_name}/{output_split}...")
        
        # Save as jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset[hf_split]:
                # Convert to ANLI format
                entry = {
                    'context': item['premise'],
                    'hypothesis': item['hypothesis'],
                    'label': ['e', 'n', 'c'][item['label']],
                    'uid': item.get('uid', '')
                }
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved: {output_file}")
    
    print("\nANLI download complete")
    print(f"Downloaded to: {base_dir}")

if __name__ == "__main__":
    download_anli()