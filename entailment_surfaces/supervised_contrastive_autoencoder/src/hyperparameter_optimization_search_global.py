"""
Coarse hyperparameter search for contrastive autoencoder
Focus on key hyperparameters with good logging for manual review
"""

import sys
import os
import json
import itertools
import shutil
from pathlib import Path
from datetime import datetime

# Add the src directory to path
sys.path.append('src')

from full_pipeline_global import main, create_experiment_config

class CoarseHyperparameterSearch:
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_accuracy = 0.0
        self.best_experiment_dir = None  # Track best experiment directory
        
    def run_coarse_search(self):
        """
        Coarse grid search over key hyperparameters
        Excludes epochs since patience handles early stopping
        """
        # Define coarse search space - focus on classification performance
        search_space = {
            'learning_rates': [5e-5, 1e-4, 2e-4],          # Current is 1e-4, test one above/below
            'latent_dims': [50, 75, 100],                   # Current is 75, test around it
            'batch_sizes': [510, 1020, 1536]                # Around current 1020
        }
        
        total_combinations = (len(search_space['learning_rates']) * 
                            len(search_space['latent_dims']) * 
                            len(search_space['batch_sizes']))
        
        print("="*70)
        print("COARSE HYPERPARAMETER SEARCH - CONCAT EMBEDDINGS")
        print("="*70)
        print(f"Baseline accuracy: 81.67% (concat embeddings)")
        print(f"Embedding type: concat (best performing from initial results)")
        print(f"Focus: Classification performance optimization")
        print(f"Reconstruction weight: Fixed at 0.3 (optimize classification first)")
        print(f"Total combinations to test: {total_combinations}")
        print(f"Search space:")
        for param, values in search_space.items():
            print(f"  {param}: {values}")
        print("="*70)
        
        experiment_count = 0
        
        for lr, latent_dim, batch_size in itertools.product(
            search_space['learning_rates'],
            search_space['latent_dims'], 
            search_space['batch_sizes']
        ):
            experiment_count += 1
            config_name = f"coarse_lr{lr:.0e}_lat{latent_dim}_bs{batch_size}"
            
            print(f"\n[{experiment_count}/{total_combinations}] Testing: {config_name}")
            print(f"  Learning rate: {lr}")
            print(f"  Latent dim: {latent_dim}")  
            print(f"  Batch size: {batch_size}")
            print(f"  Reconstruction weight: 0.3 (fixed)")
            
            try:
                accuracy = self._run_single_experiment(
                    learning_rate=lr,
                    latent_dim=latent_dim,
                    batch_size=batch_size,
                    experiment_name=config_name
                )
                
                # Track best result
                if accuracy > self.best_accuracy:
                    # New best found - save this experiment directory
                    if self.best_experiment_dir and os.path.exists(self.best_experiment_dir):
                        print(f"  Removing previous best: {os.path.basename(self.best_experiment_dir)}")
                        shutil.rmtree(self.best_experiment_dir)
                    
                    self.best_accuracy = accuracy
                    self.best_config = {
                        'learning_rate': lr,
                        'latent_dim': latent_dim,
                        'batch_size': batch_size
                    }
                    self.best_experiment_dir = exp_dir
                    print(f"  NEW BEST: {accuracy:.4f}% (improvement: +{accuracy-81.67:.2f}%)")
                    print(f"  Saved best experiment: {os.path.basename(exp_dir)}")
                else:
                    # Not the best - clean up this experiment directory
                    print(f"  Result: {accuracy:.4f}% (improvement: +{accuracy-81.67:.2f}%)")
                    print(f"  Cleaning up: {os.path.basename(exp_dir)}")
                    shutil.rmtree(exp_dir)
                
            except Exception as e:
                print(f"  FAILED: {e}")
                continue
        
        print("\n" + "="*70)
        print("COARSE SEARCH COMPLETED")
        print("="*70)
        print(f"Best accuracy: {self.best_accuracy:.4f}%")
        print(f"Total improvement: +{self.best_accuracy - 81.67:.2f}%")
        print(f"Best config: {self.best_config}")
        print(f"Best model saved in: {os.path.basename(self.best_experiment_dir) if self.best_experiment_dir else 'None'}")
        
        return self.best_config
    
    def _run_single_experiment(self, learning_rate, latent_dim, batch_size, experiment_name):
        """
        Run a single experiment with given hyperparameters
        Keep reconstruction weight fixed at 0.3 to focus on classification
        """
        config_override = {
            'data': {
                'embedding_type': 'concat',  # Use best-performing embedding type
                'batch_size': batch_size
            },
            'model': {
                'latent_dim': latent_dim
            },
            'optimizer': {
                'lr': learning_rate
            },
            'training': {
                'num_epochs': 300,  # High ceiling, let patience stop it
                'patience': 15,     # Reasonable patience for early stopping
                'save_every': 20    # Less frequent saves to speed up
            },
            'output': {
                'experiment_name': experiment_name,
                'save_plots': False  # Skip plots to save time and storage
            }
        }
        
        # Run experiment
        exp_dir, train_history, evaluation_results = main(config_override)
        
        # Extract accuracy from evaluation results
        if evaluation_results and 'classification' in evaluation_results:
            accuracy = evaluation_results['classification']['accuracy']
        else:
            raise ValueError("Failed to get classification accuracy from results")
        
        # Store result for later analysis
        result_entry = {
            'experiment_name': experiment_name,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'latent_dim': latent_dim,
                'batch_size': batch_size,
                'reconstruction_weight': 0.3  # Fixed for this search
            },
            'accuracy': accuracy,
            'improvement': accuracy - 81.67,
            'experiment_dir': exp_dir,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result_entry)
        return accuracy
    
    def save_search_results(self):
        """
        Save all search results with detailed analysis
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"coarse_hyperparameter_search_{timestamp}.json"
        
        # Sort results by accuracy for easy review
        sorted_results = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)
        
        summary = {
            'search_summary': {
                'best_accuracy': self.best_accuracy,
                'baseline_accuracy': 81.67,
                'total_improvement': self.best_accuracy - 81.67,
                'best_config': self.best_config,
                'best_experiment_dir': os.path.basename(self.best_experiment_dir) if self.best_experiment_dir else None,
                'total_experiments': len(self.results),
                'successful_experiments': len([r for r in self.results if r['accuracy'] > 0])
            },
            'top_5_results': sorted_results[:5],
            'all_results': sorted_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save a readable summary
        summary_file = f"coarse_search_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("COARSE HYPERPARAMETER SEARCH RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Baseline accuracy: 81.67%\n")
            f.write(f"Best accuracy: {self.best_accuracy:.4f}%\n")
            f.write(f"Total improvement: +{self.best_accuracy - 81.67:.2f}%\n\n")
            f.write(f"Best configuration:\n")
            for param, value in self.best_config.items():
                f.write(f"  {param}: {value}\n")
            f.write(f"\nBest model directory: {os.path.basename(self.best_experiment_dir) if self.best_experiment_dir else 'None'}\n")
            f.write(f"\nTop 10 Results:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(sorted_results[:10]):
                f.write(f"{i+1:2d}. {result['accuracy']:6.2f}% (+{result['improvement']:5.2f}%) - {result['experiment_name']}\n")
            f.write(f"\nNote: Only the best model directory is preserved to save disk space.\n")
            f.write(f"All other experiment directories were automatically cleaned up.\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Readable summary saved to: {summary_file}")
        return results_file, summary_file

def run_coarse_search():
    """
    Run coarse hyperparameter search only
    """
    searcher = CoarseHyperparameterSearch()
    
    # Run the search
    best_config = searcher.run_coarse_search()
    
    # Save results
    results_file, summary_file = searcher.save_search_results()
    
    print(f"\nSearch completed. Review {summary_file} for easy analysis.")
    print(f"Best model saved in: experiments/{os.path.basename(searcher.best_experiment_dir) if searcher.best_experiment_dir else 'None'}")
    print(f"All other experiment directories were cleaned up to save space.")

if __name__ == "__main__":
    run_coarse_search()