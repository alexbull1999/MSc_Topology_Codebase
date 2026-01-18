import os

# Set cache directory to unlimited storage location
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling - take average of all token embeddings, weighted by attention mask"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_snli_data(file_path):
    """Load SNLI data from JSON file
    
    Expected format: array of [premise, hypothesis, label]
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def filter_contradictions(data):
    """Filter to only contradiction examples
    
    Args:
        data: List of [premise, hypothesis, label] entries
    
    Returns:
        List of dicts with premise, hypothesis, and index
    """
    contradictions = []
    for idx, item in enumerate(data):
        premise, hypothesis, label = item
        if label == 'contradiction':
            contradictions.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'index': idx
            })
    return contradictions

def encode_with_sbert(contradictions, model_name='sentence-transformers/all-mpnet-base-v2', batch_size=32):
    """Encode premises and hypotheses with SBERT using transformers library"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading SBERT model: {model_name}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    premises = [item['premise'] for item in contradictions]
    hypotheses = [item['hypothesis'] for item in contradictions]
    
    # Encode premises
    print(f"Encoding {len(premises)} premises...")
    premise_embeddings = []
    for i in range(0, len(premises), batch_size):
        batch = premises[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            premise_embeddings.append(embeddings.cpu())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i+len(batch)}/{len(premises)} premises")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    premise_embeddings = torch.cat(premise_embeddings, dim=0).numpy()
    
    # Encode hypotheses
    print(f"Encoding {len(hypotheses)} hypotheses...")
    hypothesis_embeddings = []
    for i in range(0, len(hypotheses), batch_size):
        batch = hypotheses[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            hypothesis_embeddings.append(embeddings.cpu())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i+len(batch)}/{len(hypotheses)} hypotheses")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    hypothesis_embeddings = torch.cat(hypothesis_embeddings, dim=0).numpy()
    
    return premise_embeddings, hypothesis_embeddings

def main():
    # Configuration
    snli_train_path = "data/raw/snli/train/snli_10k_subset_balanced.json"
    output_path = "/vol/bitbucket/ahb24/tda_entailment_new/contradictions_only/contradiction_embeddings_SBERT_snli_10k_subset_balanced.pt"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 1: LOADING AND FILTERING SNLI DATA")
    print("=" * 80)
    
    # Check if file exists
    if not Path(snli_train_path).exists():
        print(f"\nERROR: File not found: {snli_train_path}")
        print("\nPlease update the 'snli_train_path' variable in this script to point to your SNLI train file.")
        print("Expected format: JSON array of [premise, hypothesis, label] entries")
        return
    
    print(f"\nLoading SNLI train data from: {snli_train_path}")
    snli_data = load_snli_data(snli_train_path)
    print(f"Total samples loaded: {len(snli_data)}")
    
    print("\nFiltering to contradiction examples only...")
    contradictions = filter_contradictions(snli_data)
    print(f"Contradiction samples: {len(contradictions)}")
    
    print("\n" + "=" * 80)
    print("STEP 2: ENCODING WITH SBERT")
    print("=" * 80)
    
    premise_embeddings, hypothesis_embeddings = encode_with_sbert(contradictions)
    
    # Concatenate embeddings (as used in your thesis experiments)
    concatenated_embeddings = np.concatenate([premise_embeddings, hypothesis_embeddings], axis=1)
    
    print(f"\nPremise embedding shape: {premise_embeddings.shape}")
    print(f"Hypothesis embedding shape: {hypothesis_embeddings.shape}")
    print(f"Concatenated embedding shape: {concatenated_embeddings.shape}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING ENCODED DATA")
    print("=" * 80)
    
    output_data = {
        'premise_embeddings': premise_embeddings,
        'hypothesis_embeddings': hypothesis_embeddings,
        'concatenated_embeddings': concatenated_embeddings,
        'metadata': [{'premise': c['premise'], 
                      'hypothesis': c['hypothesis'],
                      'original_index': c['index']} 
                     for c in contradictions]
    }
    
    torch.save(output_data, output_path)
    
    print(f"\nData saved to: {output_path}")
    print(f"Total contradiction samples: {len(contradictions)}")
    print(f"Embedding dimension: {concatenated_embeddings.shape[1]}")
    print("\nYou can now use this file for clustering experiments.")
    print("=" * 80)

if __name__ == "__main__":
    main()