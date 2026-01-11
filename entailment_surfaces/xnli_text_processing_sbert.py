# entailment_surfaces/text_processing_xnli_sbert.py
"""
Process XNLI combined datasets to SBERT sentence embeddings
Uses paraphrase-multilingual-mpnet-base-v2 for multilingual support
"""

import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# Set cache directory to unlimited storage
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'

class XNLITextToEmbedding:
    """Process XNLI to sentence embeddings using multilingual SBERT"""
    
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        print(f"Loading {model_name} on {self.device}...")
        
        # Set environment variable for memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
            except ValueError as e:
                print(f"Gradient checkpointing not supported for {self.model.__class__.__name__}: {e}")
        
        print("XNLI text processing pipeline ready")
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Convert list of texts to SBERT embeddings using mean pooling
        Args:
            texts (List[str]): list of texts to encode
            batch_size (int, optional): batch size. Defaults to 32.
        Returns:
            torch.Tensor: SBERT embeddings (of shape (n_texts, hidden_size))
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Move each tensor in the inputs dictionary to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # SBERT: Mean pooling instead of CLS token
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()).float()
                batch_embeddings = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Move to CPU immediately to free GPU memory
                batch_embeddings = batch_embeddings.cpu()
                embeddings.append(batch_embeddings)
                
                # Clear intermediate results from GPU
                del outputs
                del inputs
            
            # Clear cache every few batches
            if (i // batch_size + 1) % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Final cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return torch.cat(embeddings, dim=0)
    
    def _analyze_labels(self, labels: List[str]) -> Dict:
        """Analyze label distribution in dataset"""
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def process_entailment_dataset(self, dataset_path: str, language: str = 'en') -> Dict:
        """Process XNLI combined dataset to embeddings
        Args:
            dataset_path: Path to JSON file with entailment pairs
            language: Language code for metadata
        Returns:
            Dict containing embeddings and metadata
        """
        print(f"Processing dataset: {dataset_path}...")
        
        # Load dataset
        with open(dataset_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract premises, hypotheses, labels
        premises = [item[0] for item in data]
        hypotheses = [item[1] for item in data]
        labels = [item[2] for item in data]
        
        print(f"Dataset contains {len(data)} premise-hypothesis pairs")
        print("Generating premise embeddings...")
        premise_embeddings = self.encode_text(premises)
        print("Generating hypothesis embeddings...")
        hypothesis_embeddings = self.encode_text(hypotheses)
        
        # Prepare output (matching original format)
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
        
        print("Dataset processing complete")
        return result
    
    def process_single_pair(self, premise: str, hypothesis: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single premise-hypothesis pair.
        Args:
            premise: premise text
            hypothesis: hypothesis text
        Returns:
            Tuple of (premise_embeddings, hypothesis_embeddings)
        """
        premise_embeddings = self.encode_text([premise])
        hypothesis_embeddings = self.encode_text([hypothesis])
        return premise_embeddings[0], hypothesis_embeddings[0]
    
    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save processed embeddings and metadata
        Args:
            processed_data (Dict): Output from process_entailment_dataset
            output_path (str): Path to save the processed data
        """
        torch.save(processed_data, output_path)
        print(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, data_path: str) -> Dict:
        """Load previously processed data"""
        data = torch.load(data_path)
        print(f"Loaded processed data from {data_path}")
        print(f"Contains {data['metadata']['n_samples']} samples")
        return data
    
    def validate_embeddings(self, processed_data: Dict):
        """Validate that embeddings are reasonable"""
        
        premise_embs = processed_data["premise_embeddings"]
        hypothesis_embs = processed_data["hypothesis_embeddings"]
        print(f"Premise embeddings shape: {premise_embs.shape}")
        print(f"Hypothesis embeddings shape: {hypothesis_embs.shape}")
        print(f"Embedding dimension: {premise_embs.shape[1]}")
        
        # Check for reasonable ranges
        print(f"Premise embedding range: [{premise_embs.min():.3f}, {premise_embs.max():.3f}]")
        print(f"Hypothesis embedding range: [{hypothesis_embs.min():.3f}, {hypothesis_embs.max():.3f}]")
        
        # Check for NaN values
        assert not torch.isnan(premise_embs).any(), "NaN values in premise embeddings"
        assert not torch.isnan(hypothesis_embs).any(), "NaN values in hypothesis embeddings"

def main():
    """Process XNLI dataset for a specific language"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='en',
                       help='Language code (e.g., en, zh, ar, de, es, fr, hi, ru, sw, th, tr, ur, vi, bg, el)')
    parser.add_argument('--model', type=str, 
                       default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                       help='Model name to use for embeddings')
    args = parser.parse_args()
    
    processor = XNLITextToEmbedding(model_name=args.model)
    
    language = args.language
    
    # Process combined data
    data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/raw/xnli_{language}_combined.json"
    
    if Path(data_path).exists():
        processed_data = processor.process_entailment_dataset(data_path, language=language)
        processor.validate_embeddings(processed_data)
        
        # Create output directory
        output_dir = Path("/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/processed/")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"xnli_{language}_combined_SBERT.pt"
        processor.save_processed_data(processed_data, output_path)
        
        print(f"\nProcessing complete for {language}!")
        print(f"Output: {output_path}")
    else:
        print(f"Data not found at: {data_path}")
        print(f"Please run extract_xnli_combined.py first")

if __name__ == "__main__":
    main()