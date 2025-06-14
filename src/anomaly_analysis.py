import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SafeAnomalyExtractor:
    """
    Safely extracts anomalous entailment pairs by matching text content
    rather than relying on index ordering
    """

    def __init__(self, data_dir: str = "validation_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("anomaly_analysis")
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self._load_data()

    def _load_data(self):
        """Load required data files"""
        print("Loading data files...")

        # Load TDA-ready data with cone violations
        tda_data_path = self.data_dir / "tda_ready_data_SNLI_1k.pt"
        if not tda_data_path.exists():
            raise FileNotFoundError(f"TDA data not found at {tda_data_path}")

        self.tda_data = torch.load(tda_data_path, map_location='cpu')
        print(f"Loaded TDA data: {len(self.tda_data['cone_violations'])} samples")

        # Load original SNLI subset with text from JSON
        self.data_dir = Path("data")
        snli_json_path = self.data_dir / "raw/snli/train/snli_1k_subset_balanced.json"
        if not snli_json_path.exists():
            raise FileNotFoundError(f"SNLI JSON not found at {snli_json_path}")

        with open(snli_json_path, 'r', encoding='utf-8') as f:
            self.snli_data = json.load(f)
        print(f"Loaded SNLI text data: {len(self.snli_data)} samples")

        # Verify we can match the data safely
        self._verify_data_matching()

    def _verify_data_matching(self):
        """
        Verify that we can safely match TDA data with SNLI text data
        """
        print("Verifying data matching...")

        # Check if we have text data in TDA results for direct matching
        if 'texts' in self.tda_data:
            print("✓ TDA data contains text - can match directly")
            self.matching_method = 'direct'
        else:
            print("⚠ TDA data doesn't contain text - will attempt index matching")
            print("  WARNING: This assumes consistent ordering between files!")

            # Check if lengths match as a basic sanity check
            if len(self.tda_data['cone_violations']) == len(self.snli_data):
                print(f"✓ Data lengths match ({len(self.snli_data)} samples)")
                self.matching_method = 'index'
            else:
                raise ValueError(
                    f"Data length mismatch: TDA={len(self.tda_data['cone_violations'])}, SNLI={len(self.snli_data)}")

    def _get_text_for_sample(self, tda_index: int) -> Tuple[str, str, str]:
        """
        Safely get premise, hypothesis, label for a TDA sample

        Args:
            tda_index: Index in the TDA data

        Returns:
            (premise, hypothesis, label)
        """
        if self.matching_method == 'direct':
            # Use text directly from TDA data if available
            texts = self.tda_data['texts'][tda_index]
            return texts['premise'], texts['hypothesis'], texts['label']
        else:
            # Use index matching with JSON data
            # Debug: Check the structure of the JSON data
            if tda_index == 0:  # Only print once for debugging
                print(f"JSON data type: {type(self.snli_data)}")
                if isinstance(self.snli_data, list) and len(self.snli_data) > 0:
                    print(f"First sample type: {type(self.snli_data[0])}")
                    print(f"First sample: {self.snli_data[0]}")
                elif isinstance(self.snli_data, dict):
                    print(f"JSON keys: {list(self.snli_data.keys())}")

            # Handle different JSON structures
            if isinstance(self.snli_data, list):
                # If it's a list of samples
                sample = self.snli_data[tda_index]
                if isinstance(sample, dict):
                    return sample['premise'], sample['hypothesis'], sample['label']
                elif isinstance(sample, list) and len(sample) >= 3:
                    return sample[0], sample[1], sample[2]  # [premise, hypothesis, label]
                else:
                    raise ValueError(f"Unexpected sample format: {type(sample)}")
            elif isinstance(self.snli_data, dict):
                # If it's a dict with keys containing lists
                # Try common structures
                if 'data' in self.snli_data:
                    sample = self.snli_data['data'][tda_index]
                elif 'samples' in self.snli_data:
                    sample = self.snli_data['samples'][tda_index]
                else:
                    # Try to find the actual data
                    for key, value in self.snli_data.items():
                        if isinstance(value, list) and len(value) == 990:
                            sample = value[tda_index]
                            break
                    else:
                        raise ValueError(f"Could not find sample data in JSON structure")

                if isinstance(sample, dict):
                    return sample['premise'], sample['hypothesis'], sample['label']
                elif isinstance(sample, list) and len(sample) >= 3:
                    return sample[0], sample[1], sample[2]
                else:
                    raise ValueError(f"Unexpected sample format: {type(sample)}")
            else:
                raise ValueError(f"Unexpected JSON data type: {type(self.snli_data)}")

    def identify_anomalous_entailment_pairs(self, cone_energy_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Identify entailment pairs with high cone violation energy

        Args:
            cone_energy_threshold: Threshold for anomalous cone energy

        Returns:
            List of anomalous examples in original data format
        """
        print(f"\nIdentifying anomalous entailment pairs...")
        print(f"Cone energy threshold: {cone_energy_threshold}")

        anomalous_examples = []

        # Debug: Check the structure of TDA data
        print("Debugging TDA data structure:")
        print(f"TDA data keys: {list(self.tda_data.keys())}")
        if 'labels' in self.tda_data:
            labels = self.tda_data['labels']
            print(f"Labels type: {type(labels)}")
            print(f"First few labels: {labels[:5] if hasattr(labels, '__getitem__') else 'Cannot slice'}")

        # Handle different label formats
        if 'labels' in self.tda_data:
            labels = self.tda_data['labels']
            # Convert tensor to list if needed
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
        else:
            print("No 'labels' key found in TDA data")
            return []

        # Find entailment indices
        entailment_indices = []
        for i, label in enumerate(labels):
            if str(label).lower() == 'entailment' or label == 0:  # Handle both string and numeric labels
                entailment_indices.append(i)

        print(f"Found {len(entailment_indices)} entailment pairs out of {len(labels)} total")

        for tda_idx in entailment_indices:
            try:
                # Calculate cone violation energy
                cone_violations = self.tda_data['cone_violations'][tda_idx]
                cone_energy = float(torch.norm(cone_violations))

                # Check if anomalous
                if cone_energy > cone_energy_threshold:
                    # Get original text data
                    premise, hypothesis, label = self._get_text_for_sample(tda_idx)

                    # Create example in original format
                    example = {
                        'premise': premise,
                        'hypothesis': hypothesis,
                        'label': label,
                        'metadata': {
                            'tda_index': tda_idx,
                            'cone_energy': cone_energy,
                            'cone_violations': cone_violations.tolist(),
                            'anomaly_threshold': cone_energy_threshold,
                            'anomaly_ratio': cone_energy / cone_energy_threshold
                        }
                    }

                    anomalous_examples.append(example)

            except Exception as e:
                print(f"Warning: Could not process sample {tda_idx}: {e}")
                # Debug: Print more info about the failing sample
                print(
                    f"  Label at index {tda_idx}: {labels[tda_idx] if tda_idx < len(labels) else 'INDEX OUT OF RANGE'}")
                continue

        print(f"Found {len(anomalous_examples)} anomalous entailment pairs")

        # Sort by cone energy (highest first)
        anomalous_examples.sort(key=lambda x: x['metadata']['cone_energy'], reverse=True)

        return anomalous_examples

    def run_analysis(self, cone_energy_threshold: float = 0.2):
        """
        Run complete analysis and export results

        Args:
            cone_energy_threshold: Threshold for identifying anomalies

        Returns:
            Path to exported JSON file
        """
        print("=" * 60)
        print("SNLI ENTAILMENT ANOMALY EXTRACTION")
        print("=" * 60)

        # Identify anomalous pairs
        anomalous_examples = self.identify_anomalous_entailment_pairs(cone_energy_threshold)


        output_path = self.output_dir / "anomalous_entailment_pairs.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anomalous_examples, f, indent=2, ensure_ascii=False)

        # Print summary
        print("\n" + "=" * 40)
        print("ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Total entailment pairs: 330")
        print(f"Anomalous pairs found: {len(anomalous_examples)}")
        print(f"Potential mislabeling rate: {len(anomalous_examples) / 330 * 100:.1f}%")
        print(f"Highest cone energy: {max(ex['metadata']['cone_energy'] for ex in anomalous_examples):.4f}")
        print(f"Lowest anomalous cone energy: {min(ex['metadata']['cone_energy'] for ex in anomalous_examples):.4f}")

        print(f"\nExample of most anomalous pair:")
        top_example = anomalous_examples[0]
        print(f"  Premise: \"{top_example['premise'][:100]}...\"")
        print(f"  Hypothesis: \"{top_example['hypothesis'][:100]}...\"")
        print(f"  Cone Energy: {top_example['metadata']['cone_energy']:.4f}")



def main():
    """
    Main function to run the anomaly extraction
    """
    # Initialize extractor
    extractor = SafeAnomalyExtractor()

    # Run analysis with adjustable threshold
    # Start with a lower threshold to catch more potential anomalies
    extractor.run_analysis(cone_energy_threshold=0.5)

    print(f"\nAnalysis complete!")
    print("\nNext steps:")
    print("1. Review the JSON file to examine anomalous pairs")
    print("2. Manually inspect premise-hypothesis pairs for mislabeling")
    print("3. Adjust threshold if needed and re-run")



if __name__ == "__main__":
    main()