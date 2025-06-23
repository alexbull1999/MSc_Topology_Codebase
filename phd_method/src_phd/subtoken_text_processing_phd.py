import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from typing import List, Tuple, Dict, Optional
import numpy as np

class SubtokenTextToEmbedding:
    """
    Subtoken-level text processing for PHD computation using token-level embeddings
    instead of just CLS tokens. Creates rich point clouds for individual samples.
    
    This replaces the CLS-only approach in text_processing_phd.py to enable
    meaningful PHD computation on individual premise-hypothesis pairs.
    """

    def __init__(self, model_name="roberta-base", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_text_tokens(self, texts: List[str], batch_size: int = 16, include_special_tokens: bool=True) -> List[torch.Tensor]:
        """
        Convert list of texts to token-level embeddings for PHD computation
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            include_special_tokens: Whether to include [CLS], [SEP] tokens
            
        Returns:
            List of tensors, each containing all token embeddings for one text
            Each tensor has shape [n_tokens, hidden_size]
        """

        all_token_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            #Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                return_attention_mask=True
            )

            #move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            #Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                for j in range(token_embeddings.shape[0]):
                    sample_tokens = token_embeddings[j]
                    sample_mask = attention_mask[j]

                    #Filter out padding tokens
                    valid_tokens = sample_tokens[sample_mask.bool()]

                    #Optionally remove special tokens
                    if not include_special_tokens:
                        if valid_tokens.shape[0] > 2:
                            valid_tokens = valid_tokens[1:-1]

                    all_token_embeddings.append(valid_tokens)

        return all_token_embeddings


    def create_premise_hypothesis_pointcloud(self, premise: str, hypothesis: str, concatenation_method: str = "sequential") -> torch.Tensor:
        """
        Create a point cloud from premise-hypothesis pair using token embeddings
        
        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            concatenation_method: How to combine premise and hypothesis tokens
                - "sequential": premise tokens + hypothesis tokens
                - "interleaved": alternate between premise and hypothesis tokens
                - "embedded_concat": concatenate embeddings at each position
                
        Returns:
            Tensor of shape [total_tokens, hidden_size] representing the point cloud
        """

        premise_tokens = self.encode_text_tokens([premise])[0]
        hypothesis_tokens = self.encode_text_tokens([hypothesis])[0]

        if concatenation_method == "sequential":
            pointcloud = torch.cat([premise_tokens, hypothesis_tokens], dim=0)

        elif concatenation_method == "interleaved":
            # Interleave tokens from premise and hypothesis
            min_len = min(premise_tokens.shape[0], hypothesis_tokens.shape[0])
            max_len = max(premise_tokens.shape[0], hypothesis_tokens.shape[0])
            
            interleaved = []
            for i in range(min_len):
                interleaved.append(premise_tokens[i])
                interleaved.append(hypothesis_tokens[i])
            
            # Add remaining tokens from longer sequence
            if premise_tokens.shape[0] > min_len:
                interleaved.extend([premise_tokens[i] for i in range(min_len, premise_tokens.shape[0])])
            elif hypothesis_tokens.shape[0] > min_len:
                interleaved.extend([hypothesis_tokens[i] for i in range(min_len, hypothesis_tokens.shape[0])])
                
            pointcloud = torch.stack(interleaved)
            
        elif concatenation_method == "embedded_concat":
            # Concatenate embeddings dimension-wise at each token position
            min_len = min(premise_tokens.shape[0], hypothesis_tokens.shape[0])
            
            concatenated_tokens = []
            for i in range(min_len):
                # Concatenate premise and hypothesis embeddings at position i
                concat_embedding = torch.cat([premise_tokens[i], hypothesis_tokens[i]], dim=0)
                concatenated_tokens.append(concat_embedding)
            
            pointcloud = torch.stack(concatenated_tokens)
            
        else:
            raise ValueError(f"Unknown concatenation method: {concatenation_method}")
            
        return pointcloud

    
    def process_entailment_dataset_subtokens(self, dataset_path: str, concatenation_method: str = "sequential", include_class_separation: bool = True) -> Dict:
        """
        Process entailment dataset to subtoken-level point clouds for PHD computation
        
        Args:
            dataset_path: Path to JSON file with entailment pairs
            concatenation_method: How to combine premise/hypothesis tokens
            include_class_separation: Whether to organize by class
            
        Returns:
            Dict containing point clouds and metadata for PHD computation
        """
        print(f"Processing dataset with subtoken-level embeddings: {dataset_path}...")

        # Load dataset
        with open(dataset_path, "r") as file:
            data = json.load(file)

        premises = [item[0] for item in data]
        hypotheses = [item[1] for item in data]
        labels = [item[2] for item in data]

        print(f"Dataset contains {len(data)} premise-hypothesis pairs")
        print(f"Using concatenation method: {concatenation_method}")

        # Create point clouds for each sample
        pointclouds = []
        pointcloud_sizes = []
        
        for i in range(len(premises)):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(data)}")

            current_premise=premises[i]
            current_hypothesis = hypotheses[i]
                
            pointcloud = self.create_premise_hypothesis_pointcloud(
                current_premise, current_hypothesis, concatenation_method
            )
            pointclouds.append(pointcloud)
            pointcloud_sizes.append(pointcloud.shape[0])

        print(f"Point cloud statistics:")
        print(f"  Average points per sample: {np.mean(pointcloud_sizes):.1f}")
        print(f"  Min points: {np.min(pointcloud_sizes)}")
        print(f"  Max points: {np.max(pointcloud_sizes)}")
        print(f"  Embedding dimension: {pointclouds[0].shape[1]}")

        # Prepare result
        result = {
            "pointclouds": pointclouds,
            "pointcloud_sizes": pointcloud_sizes,
            "labels": labels,
            "texts": {
                "premises": premises,
                "hypotheses": hypotheses
            },
            "metadata": {
                "model_name": self.model_name,
                "concatenation_method": concatenation_method,
                "embedding_dim": pointclouds[0].shape[1],
                "n_samples": len(data),
                "avg_points_per_sample": float(np.mean(pointcloud_sizes)),
                "min_points": int(np.min(pointcloud_sizes)),
                "max_points": int(np.max(pointcloud_sizes)),
            }
        }

        if include_class_separation:
            class_pointclouds = self.organize_pointclouds_by_class(pointclouds, labels)
            result["class_pointclouds"] = class_pointclouds

        print("Subtoken-level dataset processing complete")
        return result

    def organize_pointclouds_by_class(self, pointclouds: List[torch.Tensor], labels: List[str]) -> Dict[str, List[torch.Tensor]]:
        """
        Organize point clouds by entailment class for PHD computation
        
        Args:
            pointclouds: List of point cloud tensors
            labels: List of entailment labels
            
        Returns:
            Dict mapping class names to lists of their point clouds
        """

        class_pointclouds = {}
        
        unique_labels = list(set(labels))
        for label in unique_labels:
            class_clouds = [pointclouds[i] for i in range(len(labels)) if labels[i] == label]
            class_pointclouds[label] = class_clouds
            
            total_points = sum(cloud.shape[0] for cloud in class_clouds)
            avg_points = total_points / len(class_clouds) if class_clouds else 0
            
            print(f"Class '{label}': {len(class_clouds)} samples, "
                  f"avg {avg_points:.1f} points per sample")

        return class_pointclouds


    def inspect_tokenization(self, text: str, max_display: int = 20):
        """
        Inspect how text gets tokenized (useful for debugging)
        
        Args:
            text: Text to inspect
            max_display: Maximum number of tokens to display
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        print(f"Text: '{text}'")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens (first {max_display}): {tokens[:max_display]}")
        print(f"Token IDs (first {max_display}): {token_ids[:max_display]}")
        
        if len(tokens) > max_display:
            print(f"... and {len(tokens) - max_display} more tokens")

    
    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save processed subtoken-level data"""
        torch.save(processed_data, output_path)
        print(f"Saved processed subtoken-level data to {output_path}")

    def validate_pointclouds(self, processed_data: Dict):
        """Validate that point clouds are reasonable for PHD computation"""
        pointclouds = processed_data["pointclouds"]
        metadata = processed_data["metadata"]
        
        print(f"Point cloud validation:")
        print(f"  Total samples: {len(pointclouds)}")
        print(f"  Average points per sample: {metadata['avg_points_per_sample']}")
        print(f"  Min/Max points: {metadata['min_points']}/{metadata['max_points']}")
        print(f"  Embedding dimension: {metadata['embedding_dim']}")
        
        # Check for minimum point requirements for PHD
        min_points = metadata['min_points']
        if min_points < 10:
            print(f"WARNING: Minimum points ({min_points}) may be too low for stable PHD")
        else:
            print(f"Minimum points ({min_points}) should be sufficient for PHD")
        
        # Check for NaN values
        nan_found = False
        for i, cloud in enumerate(pointclouds[:10]):  # Check first 10
            if torch.isnan(cloud).any():
                print(f"ERROR: NaN values found in point cloud {i}")
                nan_found = True
                break
        
        if not nan_found:
            print("No NaN values detected in sample point clouds")
            
        # Check embedding ranges
        sample_cloud = pointclouds[0]
        print(f"Sample embedding range: [{sample_cloud.min():.3f}, {sample_cloud.max():.3f}]")
        
        print("Point cloud validation complete")


def test_subtoken_processing():
    processor=SubtokenTextToEmbedding()

    #Test single pair to inspect tokenization
    premise = "A man is walking in the park."
    hypothesis = "Someone is outside."

    print("=== Testing single pair processing ===")
    print("\nPremise tokenization:")
    processor.inspect_tokenization(premise)
    print("\nHypothesis tokenization:")
    processor.inspect_tokenization(hypothesis)

    print("\n=== Processing full dataset ===")
    dataset_path="data/raw/snli/train/snli_10k_subset_balanced.json"
    output_path="phd_method/phd_data/processed/snli_10k_subset_balanced_roberta_subtokenized.pt"

    try:
        # Process the entire dataset
        print(f"Processing dataset: {dataset_path}")
        processed_data = processor.process_entailment_dataset_subtokens(
            dataset_path=dataset_path,
            concatenation_method="sequential",  # Start with sequential method
            include_class_separation=True
        )

         # Validate the processed data
        print("\n=== Validating processed data ===")
        processor.validate_pointclouds(processed_data)
        
        # Save processed data for use in PHD baseline computation
        processor.save_processed_data(processed_data, output_path)

    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {dataset_path}")
        print("Please update the dataset_path variable to point to your actual data file")
        print("Expected format: JSON file with [premise, hypothesis, label] entries")
        
    except Exception as e:
        print(f"ERROR during processing: {e}")
        print("Please check the dataset format and file path")


if __name__ == "__main__":
    test_subtoken_processing()













