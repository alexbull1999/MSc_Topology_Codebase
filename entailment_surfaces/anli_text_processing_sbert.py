import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from typing import List, Dict, Tuple
import numpy as np

# Set cache directory to unlimited storage
os.environ['HF_HOME'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/ahb24/tda_entailment_new/huggingface_cache'

class TextToEmbedding:
    """Text to embedding pipeline using BERT"""

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize text processing pipeline"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        print(f"Loading {model_name} on {device}...")

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
            except ValueError as e:
                print(f"Gradient checkpointing not supported for {self.model.__class__.__name__}: {e}")

        print("Text processing pipeline ready")

    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Convert list of texts to BERT embeddings using mean pooling"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
                batch_embeddings = batch_embeddings.cpu()
                embeddings.append(batch_embeddings)
                
                del outputs
                del inputs

            if (i // batch_size + 1) % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return torch.cat(embeddings, dim=0)

    def process_entailment_dataset(self, dataset_path: str) -> Dict:
        """Process entailment dataset to embeddings"""
        print(f"Processing dataset: {dataset_path}...")

        with open(dataset_path, "r") as file:
            data = json.load(file)

        premises = [item[0] for item in data]
        hypotheses = [item[1] for item in data]
        labels = [item[2] for item in data]

        print(f"Dataset contains {len(data)} premise-hypothesis pairs")
        print("Generating premise embeddings...")
        premise_embeddings = self.encode_text(premises)
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
                "label_counts": self._analyze_labels(labels)
            }
        }

        print("Dataset processing complete")
        return result

    def _analyze_labels(self, labels: List[str]) -> Dict:
        """Analyze label distribution in dataset"""
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save processed embeddings and metadata"""
        torch.save(processed_data, output_path)
        print(f"Saved processed data to {output_path}")

    def validate_embeddings(self, processed_data: Dict):
        """Validate that embeddings are reasonable"""
        premise_embs = processed_data["premise_embeddings"]
        hypothesis_embs = processed_data["hypothesis_embeddings"]
        print(f"Premise embeddings shape: {premise_embs.shape}")
        print(f"Hypothesis embeddings shape: {hypothesis_embs.shape}")
        print(f"Embedding dimension: {premise_embs.shape[1]}")

        print(f"Premise embedding range: [{premise_embs.min():.3f}, {premise_embs.max():.3f}]")
        print(f"Hypothesis embedding range: [{hypothesis_embs.min():.3f}, {hypothesis_embs.max():.3f}]")

        assert not torch.isnan(premise_embs).any(), "NaN values in premise embeddings"
        assert not torch.isnan(hypothesis_embs).any(), "NaN values in hypothesis embeddings"

def process_anli_round(round_name, split):
    """Process a single ANLI round and split"""
    
    processor = TextToEmbedding()
    
    data_path = f"/vol/bitbucket/ahb24/tda_entailment_new/anli_raw/{round_name}/anli_{round_name}_{split}.json"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    processed_data = processor.process_entailment_dataset(data_path)
    processor.validate_embeddings(processed_data)
    
    output_dir = "/vol/bitbucket/ahb24/tda_entailment_new/anli_processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/anli_{round_name}_{split}_SBERT.pt"
    processor.save_processed_data(processed_data, output_path)
    
    print(f"Processing complete for {round_name} {split}")

if __name__ == "__main__":
    rounds = ['R1', 'R2', 'R3']
    splits = ['train', 'dev', 'test']
    
    for round_name in rounds:
        for split in splits:
            print(f"\nProcessing ANLI {round_name} {split}...")
            process_anli_round(round_name, split)
            print()