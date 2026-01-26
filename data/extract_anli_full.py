import json
import os
from pathlib import Path

def extract_anli_full_dataset(round_dir, split, output_path, seed=42):
    """
    Extract ANLI dataset for a specific round and split
    Converts to format: [premise, hypothesis, label]
    """
    
    label_map = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}
    
    input_file = round_dir / f"{split}.jsonl"
    
    if not input_file.exists():
        print(f"File not found: {input_file}")
        return None
    
    print(f"Processing {input_file}...")
    
    output_data = []
    label_counts = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            premise = item['context']
            hypothesis = item['hypothesis']
            label = label_map.get(item['label'], item['label'])
            
            output_data.append([premise, hypothesis, label])
            
            if label in label_counts:
                label_counts[label] += 1
    
    print(f"Total samples: {len(output_data)}")
    print(f"Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    print(f"Sample format:")
    print(f"First example: {output_data[0]}")
    print(f"Expected format: [premise, hypothesis, label]")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(output_data)} samples to {output_path}")
    return output_data

def extract_all_anli():
    """Extract all ANLI rounds and splits"""
    
    base_dir = Path("/vol/bitbucket/ahb24/tda_entailment_new/anli_raw")
    rounds = ['R1', 'R2', 'R3']
    splits = ['train', 'dev', 'test']
    
    for round_name in rounds:
        round_dir = base_dir / round_name
        
        for split in splits:
            output_path = round_dir / f"anli_{round_name}_{split}.json"
            extract_anli_full_dataset(round_dir, split, output_path)
            print()

if __name__ == "__main__":
    extract_all_anli()