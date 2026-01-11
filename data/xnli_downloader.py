# data/xnli_downloader.py
from datasets import load_dataset
import json
import os
from pathlib import Path

# Set cache directory to unlimited storage location
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'

def download_xnli():
    """Download XNLI dataset from HuggingFace"""
    
    # Create cache directory
    cache_dir = Path('/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directory
    output_dir = Path("vol/bitbucket/ahb24/tda_entailment_new/xnli_data/raw_xnli")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download XNLI dataset (all languages) with explicit cache dir
    print("Downloading XNLI dataset...")
    dataset = load_dataset(
        "xnli", 
        "all_languages",
        cache_dir=str(cache_dir)
    )
    
    print(f"Available splits: {dataset.keys()}")
    
    # Check the structure
    print(f"\nDataset structure:")
    print(f"Train features: {dataset['train'].features}")
    print(f"\nFirst train example:")
    print(dataset['train'][0])
    
    return dataset

if __name__ == "__main__":
    dataset = download_xnli()
    print(f"\nTrain samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")