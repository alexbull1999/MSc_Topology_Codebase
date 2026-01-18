# entailment_surfaces/text_processing_xnli_llm2vec.py
"""
XNLI Text to Embedding Pipeline using LLM2Vec-Mistral-7B
Processes multilingual XNLI data with memory-efficient settings
"""

import os
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache/'

import torch
from llm2vec import LLM2Vec
import json
from typing import List, Dict
import argparse
from pathlib import Path
import gc

class XNLILLMtoVecTextToEmbedding:
    """Text to embedding pipeline for XNLI using LLM2Vec-Mistral-7B"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize with memory-efficient settings"""
        self.device = torch.device(device)
        print(f"Loading LLM2Vec-Mistral-7B model on {device} (memory-efficient mode)...")

        base_model = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
        peft_model = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised"
        self.model_name = "LLM2Vec-Mistral-7B-Supervised"

        # Memory-efficient loading with 8-bit quantization
        self.model = LLM2Vec.from_pretrained(
            base_model,
            peft_model_name_or_path=peft_model,
            pooling_mode="mean",
            max_length=256,  # Reduced from 512
            torch_dtype=torch.float16,
            device_map="auto",            
            low_cpu_mem_usage=True
        )
        
        print("LLM2Vec-Mistral loaded successfully (8-bit quantization)")
        print(f"Model: {self.model_name}")

    def encode_text(self, texts: List[str], batch_size: int = 2) -> torch.Tensor:
        """Encode with very small batch size for memory efficiency"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch)
                
                if not isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = torch.tensor(batch_embeddings)
                    
                batch_embeddings = batch_embeddings.cpu().float()
                all_embeddings.append(batch_embeddings)

            # Clear cache after EVERY batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)

    def process_entailment_dataset(self, json_path: str, language: str) -> Dict:
        """Process XNLI dataset for a specific language"""
        print(f"Processing XNLI dataset for language: {language}")
        print(f"Reading from: {json_path}")

        with open(json_path, "r", encoding='utf-8') as file:
            data = json.load(file)

        premises = [item[0] for item in data]
        hypotheses = [item[1] for item in data]
        labels = [item[2] for item in data]

        print(f"Dataset contains {len(data)} premise-hypothesis pairs")
        
        print("Generating premise embeddings...")
        premise_embeddings = self.encode_text(premises)
        
        # Clear memory between premise and hypothesis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("Generating hypothesis embeddings...")
        hypothesis_embeddings = self.encode_text(hypotheses)

        result = {
            "premise_embeddings": premise_embeddings,
            "hypothesis_embeddings": hypothesis_embeddings,
            "labels": labels,
            "texts": {
                "premises": premises,
                "hypotheses": hypotheses
            },
            "metadata": {
                "model_name": self.model_name,
                "embedding_dim": premise_embeddings.shape[1],
                "n_samples": len(data),
                "language": language,
                "label_counts": self._analyze_labels(labels)
            }
        }

        print(f"Dataset processing complete for {language}")
        print(f"  Embedding dimension: {premise_embeddings.shape[1]}")
        print(f"  Label distribution: {result['metadata']['label_counts']}")
        
        return result

    def _analyze_labels(self, labels: List[str]) -> Dict:
        """Analyze label distribution in dataset"""
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save processed embeddings and metadata"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(processed_data, output_path)
        print(f"Saved processed data to {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024**2):.1f} MB")

    def validate_embeddings(self, processed_data: Dict):
        """Validate that embeddings are reasonable"""
        premise_embs = processed_data["premise_embeddings"]
        hypothesis_embs = processed_data["hypothesis_embeddings"]
        
        print(f"\nValidation:")
        print(f"  Premise embeddings shape: {premise_embs.shape}")
        print(f"  Hypothesis embeddings shape: {hypothesis_embs.shape}")
        print(f"  Embedding dimension: {premise_embs.shape[1]}")
        print(f"  Premise range: [{premise_embs.min():.3f}, {premise_embs.max():.3f}]")
        print(f"  Hypothesis range: [{hypothesis_embs.min():.3f}, {hypothesis_embs.max():.3f}]")

        assert not torch.isnan(premise_embs).any(), "NaN values in premise embeddings"
        assert not torch.isnan(hypothesis_embs).any(), "NaN values in hypothesis embeddings"
        print("  ✓ No NaN values detected")


def main():
    """Process XNLI dataset for specific language(s)"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='all',
                       help='Language code (en, zh, ar, etc.) or "all" for all languages')
    args = parser.parse_args()
    
    all_languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    
    if args.language == 'all':
        languages_to_process = all_languages
        print(f"Processing all {len(all_languages)} languages")
    else:
        if args.language not in all_languages:
            print(f"ERROR: Unknown language '{args.language}'")
            print(f"Available languages: {', '.join(all_languages)}")
            return
        languages_to_process = [args.language]
        print(f"Processing single language: {args.language}")
    
    # DO NOT initialize processor here - do it per language to prevent memory buildup
    
    successful = 0
    failed = 0
    
    for idx, language in enumerate(languages_to_process, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING LANGUAGE {idx}/{len(languages_to_process)}: {language.upper()}")
        print(f"{'='*60}")
        
        try:
            # Clear memory before each language
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Initialize processor fresh for each language
            processor = XNLILLMtoVecTextToEmbedding()
            
            data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/raw/xnli_{language}_combined.json"
            
            if not Path(data_path).exists():
                print(f"ERROR: Data not found at: {data_path}")
                failed += 1
                continue
            
            processed_data = processor.process_entailment_dataset(data_path, language=language)
            processor.validate_embeddings(processed_data)
            
            output_dir = Path("/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/processed/llm2vec_mistral/")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"xnli_{language}_combined_LLM2Vec_Mistral.pt"
            processor.save_processed_data(processed_data, output_path)
            
            print(f"\n✓ {language.upper()} processing complete!")
            successful += 1
            
            # Delete everything and clear memory
            del processor
            del processed_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n✗ {language.upper()} processing FAILED!")
            print(f"Error: {e}")
            failed += 1
            
            # Clear memory even on failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(languages_to_process)}")
    print(f"Failed: {failed}/{len(languages_to_process)}")
    
    if failed > 0:
        print("\n⚠ Some languages failed to process")
        exit(1)
    else:
        print("\n✓ All languages processed successfully!")
        exit(0)


if __name__ == "__main__":
    main()