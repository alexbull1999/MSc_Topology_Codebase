"""
Integration Script for Supervised Contrastive Autoencoder
End-to-end pipeline for training and evaluating the model on SNLI data
"""

import torch
import torch.optim as optim
import os
import json
import argparse
from datetime import datetime

# Import our modules
from contrastive_autoencoder_model import ContrastiveAutoencoder
from losses import CombinedLoss
from data_loader import EntailmentDataLoader
from trainer import ContrastiveAutoencoderTrainer
from evaluator import ContrastiveAutoencoderEvaluator


def create_experiment_config():
    """Create experiment configuration"""
    config = {
        # Data configuration
        'data': {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'sample_size': None,  # Use full dataset, set to int for testing
            'batch_size_generation': 1000,  # For lattice embedding generation
            'batch_size_training': 32       # For model training
        },
        
        # Model configuration
        'model': {
            'input_dim': 768,
            'latent_dim': 75,  # Based on your PH-dim analysis
            'hidden_dims': [512, 256],
            'dropout_rate': 0.2
        },
        
        # Loss configuration
        'loss': {
            'contrastive_weight': 1.0,    # Maximum alpha parameter
            'reconstruction_weight': 1.0,
            'temperature': 0.1,
            'reconstruction_type': 'mse',
            'beta_scheduling': {
                'enabled': True,
                'warmup_epochs': 10,      # Pure reconstruction phase
                'max_beta': 2.0,          # Maximum contrastive weight
                'schedule_type': 'linear' # 'linear', 'cosine', 'exponential'
            }
        },
        
        # Optimizer configuration
        'optimizer': {
            'lr': 0.001,
            'weight_decay': 1e-4
        },
        
        # Training configuration
        'training': {
            'num_epochs': 50,
            'patience': 10,
            'save_every': 5
        },
        
        # Evaluation configuration
        'evaluation': {
            'phase1_group_size': 1000,
            'phase1_num_groups': 10
        },
        
        # Output configuration
        'output': {
            'experiment_name': f'contrastive_autoencoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'save_dir': 'entailment_surfaces/supervised_contrastive_autoencoder/experiments',
            'save_results': True,
            'save_models': True
        }
    }
    
    return config

def setup_experiment(config):
    """Setup experiment directories and save configuration"""
    # Create experiment directory
    exp_dir = os.path.join(config['output']['save_dir'], config['output']['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    results_dir = os.path.join(exp_dir, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment setup complete: {exp_dir}")
    return exp_dir, checkpoints_dir, results_dir


def load_data(config):
    """Load and prepare SNLI data"""
    print("Loading SNLI data...")
    print("=" * 50)
    
    data_loader = EntailmentDataLoader(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data']['test_path'],
        sample_size=config['data']['sample_size'],
        batch_size=config['data']['batch_size_generation']
    )
    
    # Load and process data
    train_dataset, val_dataset, test_dataset = data_loader.load_data()
    
    # Create PyTorch DataLoaders
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=config['data']['batch_size_training'],
        shuffle=True
    )
    
    print(f"Data loading completed!")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def create_model_and_trainer(config, device):
    """Create model, loss function, optimizer, and trainer"""
    print("Creating model and trainer...")
    print("=" * 50)
    
    # Create model
    model = ContrastiveAutoencoder(**config['model'])
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # This excludes beta_scheduling from the loss function initialization
    loss_config = {k: v for k, v in config['loss'].items() if k != 'beta_scheduling'}
    loss_function = CombinedLoss(**loss_config)
    print(f"Loss function created (α={config['loss']['contrastive_weight']})")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), **config['optimizer'])
    print(f"Optimizer created (lr={config['optimizer']['lr']})")
    
    # Create trainer
    trainer = ContrastiveAutoencoderTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device
    )
    
    return model, trainer


def train_model(trainer, train_loader, val_loader, config, checkpoints_dir):
    """Train the contrastive autoencoder"""
    print("Starting model training...")
    print("=" * 50)

    # Extract beta config if present
    beta_config = config['loss'].get('beta_scheduling', None)
    if beta_config:
        beta_config['total_epochs'] = config['training']['num_epochs']  # ADD THIS LINE
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        save_dir=checkpoints_dir,
        save_every=config['training']['save_every'],
        beta_config=beta_config  # ADD THIS LINE
    )
    
    print(f"Training completed!")
    print(f"Best model saved at epoch {trainer.best_epoch}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    
    return trainer.train_history


def evaluate_model(model, train_loader, val_loader, test_loader, config, results_dir, device):
    """Comprehensive model evaluation"""
    print("Starting model evaluation...")
    print("=" * 50)
    
    # Create evaluator
    evaluator = ContrastiveAutoencoderEvaluator(model, device)
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.comprehensive_evaluation(
        train_dataloader=train_loader,
        val_dataloader=val_loader, 
        test_dataloader=test_loader
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    if config['output']['save_results']:
        results_path = evaluator.save_evaluation_results(results_dir)
        print(f"Evaluation results saved to: {results_path}")
    
    return evaluation_results


def save_final_results(config, train_history, evaluation_results, exp_dir):
    """Save final experiment results"""
    print("Saving final experiment results...")
    
    final_results = {
        'experiment_config': config,
        'training_history': train_history,
        'evaluation_results': evaluation_results,
        'experiment_metadata': {
            'completion_time': datetime.now().isoformat(),
            'experiment_directory': exp_dir
        }
    }
    
    # Save comprehensive results
    final_results_path = os.path.join(exp_dir, 'final_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Final results saved to: {final_results_path}")
    
    # Print key metrics
    print("\nKEY RESULTS SUMMARY")
    print("=" * 50)
    
    if 'clustering_individual' in evaluation_results:
        clustering = evaluation_results['clustering_individual']
        print(f"Individual Clustering Accuracy: {clustering['clustering_accuracy']:.4f}")
    
    if 'clustering_phase1_style' in evaluation_results:
        phase1 = evaluation_results['clustering_phase1_style']
        if 'error' not in phase1:
            phase1_acc = phase1['centroid_clustering']['clustering_accuracy']
            print(f"Phase 1 Style Clustering Accuracy: {phase1_acc:.4f}")
    
    if 'classification' in evaluation_results:
        classification = evaluation_results['classification']
        print(f"Classification Accuracy: {classification['accuracy']:.4f}")
    
    if 'reconstruction' in evaluation_results:
        reconstruction = evaluation_results['reconstruction']
        print(f"Reconstruction MSE: {reconstruction['average_mse']:.6f}")
    
    return final_results_path


def deep_merge_configs(base_config, override_config):
    """Deep merge two configuration dictionaries"""
    if override_config is None:
        return base_config
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def main(config_override=None):
    """Main execution function"""
    print("SUPERVISED CONTRASTIVE AUTOENCODER - FULL PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create configuration
    config = create_experiment_config()
    if config_override:
        config = deep_merge_configs(config, config_override)
    
    # Setup experiment
    exp_dir, checkpoints_dir, results_dir = setup_experiment(config)
    
    try:
        # Load data
        train_loader, val_loader, test_loader = load_data(config)
        
        # Create model and trainer
        model, trainer = create_model_and_trainer(config, device)
        
        # Train model
        train_history = train_model(trainer, train_loader, val_loader, config, checkpoints_dir)
        
        # Load best model for evaluation
        best_model_path = os.path.join(checkpoints_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        evaluation_results = evaluate_model(
            model, train_loader, val_loader, test_loader, config, results_dir, device
        )
        
        # Save final results
        final_results_path = save_final_results(config, train_history, evaluation_results, exp_dir)
        
        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Experiment directory: {exp_dir}")
        print(f"Final results: {final_results_path}")
        
        return {
            'success': True,
            'experiment_dir': exp_dir,
            'final_results_path': final_results_path,
            'results': evaluation_results
        }
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'experiment_dir': exp_dir
        }


def quick_test():
    """Quick test with small dataset"""
    print("Running quick test with small dataset...")
    
    config_override = {
        'data': {
            'sample_size': 1000,  # Small dataset for testing
            'batch_size_training': 16
        },
        'training': {
            'num_epochs': 3,
            'patience': 2
        }
    }
    
    return main(config_override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run supervised contrastive autoencoder pipeline')
    parser.add_argument('--test', action='store_true', help='Run quick test with small dataset')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    
    args = parser.parse_args()
    
    if args.test:
        result = quick_test()
    else:
        config_override = None
        if args.config:
            with open(args.config, 'r') as f:
                config_override = json.load(f)
        
        result = main(config_override)
    
    if result['success']:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed!")
        exit(1)