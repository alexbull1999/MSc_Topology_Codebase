# phd_method/src_phd/xnli_sbert_token_extractor.py
"""
XNLI SBERT Token Extractor
Extracts token-level embeddings from multilingual SBERT for XNLI premise-hypothesis pairs
Processes validation and test splits separately for proper experiment protocol
"""

import os

# CRITICAL: Set cache directories BEFORE importing anything from HuggingFace
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'

import torch
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle
import gc
import argparse


class XNLISBERTTokenExtractor:
    """Extract token-level embeddings from multilingual SBERT hidden layers"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        
        print(f"Loading multilingual SBERT model: {model_name}")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
  
        self.hidden_size = self.model.config.hidden_size
        print(f"Hidden size: {self.hidden_size}")
        print(f"Multilingual SBERT token extractor ready")
    
    def extract_token_embeddings(self, text: str, max_length: int = 256) -> torch.Tensor:
        """
        Extract token-level embeddings from SBERT's last hidden layer
        
        Args:
            text: Input text (any of 15 XNLI languages)
            max_length: Maximum sequence length
            
        Returns:
            Token embeddings [num_tokens, hidden_size]
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]
            attention_mask = inputs['attention_mask'][0]
            
            valid_tokens = token_embeddings[attention_mask.bool()]
            
        return valid_tokens.cpu()
    
    def process_premise_hypothesis_pair(self, premise: str, hypothesis: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a premise-hypothesis pair to extract token embeddings"""
        premise_tokens = self.extract_token_embeddings(premise)
        hypothesis_tokens = self.extract_token_embeddings(hypothesis)
        
        return premise_tokens, hypothesis_tokens
    
    def process_xnli_split(self, data_path: str, output_path: str, language: str, 
                          split_name: str, max_samples: int = None):
        """
        Process XNLI dataset split and save token embeddings
        
        Args:
            data_path: Path to XNLI JSON dataset file
            output_path: Path to save processed embeddings
            language: Language code (e.g., 'en', 'zh', 'ar')
            split_name: Split name ('validation' or 'test')
            max_samples: Maximum number of samples to process (None for all)
        """
        print(f"Processing XNLI {split_name} split for {language}...")
        print(f"Data path: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
            print(f"Processing first {max_samples} samples")
        
        print(f"Total samples to process: {len(data)}")
        
        processed_data = {
            'premise_tokens': [],
            'hypothesis_tokens': [],
            'labels': [],
            'token_counts': {'premise': [], 'hypothesis': []},
            'metadata': {
                'model_name': self.model_name,
                'hidden_size': self.hidden_size,
                'language': language,
                'split': split_name,
                'total_samples': len(data)
            }
        }
        
        for i, (premise, hypothesis, label) in enumerate(data):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} samples")
            
            try:
                premise_tokens, hypothesis_tokens = self.process_premise_hypothesis_pair(premise, hypothesis)
                
                processed_data['premise_tokens'].append(premise_tokens)
                processed_data['hypothesis_tokens'].append(hypothesis_tokens)  
                processed_data['labels'].append(label)
                processed_data['token_counts']['premise'].append(premise_tokens.shape[0])
                processed_data['token_counts']['hypothesis'].append(hypothesis_tokens.shape[0])
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
            
            if (i + 1) % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Saving processed data to: {output_path}")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Print statistics
        premise_counts = processed_data['token_counts']['premise']
        hypothesis_counts = processed_data['token_counts']['hypothesis']
        
        print(f"\nProcessing Statistics for {language} {split_name}:")
        print(f"Successfully processed: {len(processed_data['labels'])} samples")
        print(f"Premise token counts - Mean: {np.mean(premise_counts):.1f}, "
              f"Min: {np.min(premise_counts)}, Max: {np.max(premise_counts)}")
        print(f"Hypothesis token counts - Mean: {np.mean(hypothesis_counts):.1f}, "
              f"Min: {np.min(hypothesis_counts)}, Max: {np.max(hypothesis_counts)}")
        
        label_counts = {}
        for label in processed_data['labels']:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Label distribution: {label_counts}")
        
        return processed_data


def main():
    """Process XNLI datasets with separate validation/test splits"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='all',
                       help='Language code (en, zh, ar, etc.) or "all"')
    args = parser.parse_args()
    
    all_languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    
    if args.language == 'all':
        languages_to_process = all_languages
    else:
        if args.language not in all_languages:
            print(f"ERROR: Unknown language '{args.language}'")
            return
        languages_to_process = [args.language]
    
    extractor = XNLISBERTTokenExtractor()
    
    successful = 0
    failed = 0
    
    for idx, language in enumerate(languages_to_process, 1):
        print(f"\n{'='*60}")
        print(f"LANGUAGE {idx}/{len(languages_to_process)}: {language.upper()}")
        print(f"{'='*60}")
        
        try:
            # Process validation split
            val_data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/raw_splits/xnli_{language}_validation.json"
            val_output_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/sample_level_tokens/xnli_{language}_validation_sbert_tokens.pkl"
            
            if Path(val_data_path).exists():
                print("\n--- Processing VALIDATION split ---")
                extractor.process_xnli_split(val_data_path, val_output_path, language, 'validation')
            else:
                print(f"ERROR: Validation data not found: {val_data_path}")
                failed += 1
                continue
            
            # Process test split
            test_data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/raw_splits/xnli_{language}_test.json"
            test_output_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/sample_level_tokens/xnli_{language}_test_sbert_tokens.pkl"
            
            if Path(test_data_path).exists():
                print("\n--- Processing TEST split ---")
                extractor.process_xnli_split(test_data_path, test_output_path, language, 'test')
            else:
                print(f"ERROR: Test data not found: {test_data_path}")
                failed += 1
                continue
            
            print(f"\n✓ {language.upper()} token extraction complete!")
            successful += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
                
        except Exception as e:
            print(f"\n✗ {language.upper()} FAILED: {e}")
            failed += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    print(f"\n{'='*60}")
    print(f"TOKEN EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(languages_to_process)}")
    print(f"Failed: {failed}/{len(languages_to_process)}")
    
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()