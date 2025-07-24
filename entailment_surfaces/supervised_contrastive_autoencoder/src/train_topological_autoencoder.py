import torch
import torch.optim as optim
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from full_pipeline_global import setup_experiment, load_data
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from attention_autoencoder_model import AttentionAutoencoder
from losses_global_topological import TopologicallyRegularizedCombinedLoss
from trainer_topological import TopologicalTrainer
from evaluator_global import GlobalContrastiveEvaluator


def create_topological_config():
    """
    Create configuration for topological autoencoder training.
    Based on your existing config structure but adapted for TorchPH.
    """
    config = {
        'data': {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'embedding_type': 'lattice',  # Use your best performing type
            'batch_size': 1020,
            'sample_size': None,
            'balanced_sampling': True,
            'random_state': 42
        },
        
        'model': {
            'input_dim': 1536,  
            'latent_dim': 75,
            'hidden_dims': [1024, 768, 512, 256, 128],
            'dropout_rate': 0.2
        },
        
        'loss': {
            # Phase 1: Topological + Reconstruction (NO contrastive initially)
            'contrastive_weight': 0.0,  # Start with 0
            'reconstruction_weight': 0.1,  # INCREASED: Strong semantic preservation signal
            
            # Topological loss settings
            'topological_weight': 1.00,  # Main learning signal
            'max_topological_weight': 1.00,
            'topological_warmup_epochs': 0,  # FIXED: Start immediately (no warmup)
            'prototypes_path': None,
            
            # Reconstruction scheduling (for compatibility with FullDatasetCombinedLoss)
            'schedule_reconstruction': True,  # Keep constant for Phase 1
            'warmup_epochs': 10,  # No warmup needed
            'max_reconstruction_weight': 0.3,
            'schedule_type': 'linear',
            
            # Global dataset settings (required for FullDatasetCombinedLoss compatibility)
            'margin': 2.0,  # ADDED: Required even with contrastive_weight=0
            'update_frequency': 3,
            'max_global_samples': 5000
        },
        
        
        'optimizer': {
            'lr': 0.001,  # Conservative learning rate #WAS 0.0001
            'weight_decay': 1e-5
        },
        
        'training': {
            'num_epochs': 300,  # More epochs needed for topological learning
            'patience': 20,  # More patience for topology to emerge // was 10 -- removing patience essentially
            'save_every': 5,
            'debug_frequency': 25
        },
        
        'output': {
            'save_results': True,
            'save_plots': True,
            'experiment_name': 'topological_autoencoder_torchph_phase1'
        }
    }
    
    return config


def evaluate_model(model, train_loader, val_loader, test_loader, config, results_dir, device):
    """
    Comprehensive model evaluation
    """
    print("Starting model evaluation...")
    print("=" * 40)
    
    # Create evaluator
    evaluator = GlobalContrastiveEvaluator(model, device)
    
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


def main_topological_training():
    """
    Main function for topological autoencoder training using TorchPH.
    """
    print("="*60)
    print("TOPOLOGICAL AUTOENCODER TRAINING WITH TORCHPH")
    print("="*60)
    
    # Create config
    config = create_topological_config()
    
    # Setup experiment (use your existing setup function)
    
    exp_dir, checkpoints_dir, results_dir = setup_experiment(config)
    
    # Create data loaders (reuse your existing function)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = load_data(config)
    
    # Create model (reuse your existing model)
    model = AttentionAutoencoder(**config['model'])
    
    # Create topological loss function
    loss_function = TopologicallyRegularizedCombinedLoss(**config['loss'])
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    
    # Create trainer (reuse your existing trainer)
    trainer = TopologicalTrainer(model, loss_function, optimizer, device)
    
    print("Starting Phase 1: Pure Topological Training")
    print(f"  Contrastive weight: {config['loss']['contrastive_weight']}")
    print(f"  Topological weight: {config['loss']['topological_weight']}")
    print(f"  Reconstruction weight: {config['loss']['reconstruction_weight']}")
    
    # Train model
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            patience=config['training']['patience'],
            save_dir=checkpoints_dir,
            save_every=config['training']['save_every'],
            debug_frequency=config['training']['debug_frequency']
        )
        
        print("✅ Topological training completed successfully!")
        
        # Save results
        print("Saving results...")
        # You can reuse your existing evaluation functions here

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

        # Print key results
        if evaluation_results and 'separation' in evaluation_results:
            separation = evaluation_results['separation']
            if 'error' not in separation:
                print(f"\nKey Results:")
                print(f"  Separation ratio: {separation['separation_ratio']:.2f}x")
                print(f"  Perfect separation: {separation['perfect_separation']}")
                print(f"  Classification accuracy: {evaluation_results['classification']['accuracy']:.4f}")

        # Analyze topological success
        print("\n" + "="*60)
        print("TOPOLOGICAL TRAINING ANALYSIS")
        print("="*60)
        
        # Get topological learning diagnosis
        diagnosis = trainer.diagnose_topological_progress()
        
        if diagnosis:
            topo_percentage = diagnosis['topology_percentage']
            if topo_percentage > 0.8:
                print("🚀 EXCELLENT: Consistent topological learning achieved!")
            elif topo_percentage > 0.5:
                print("✅ GOOD: Reasonable topological learning")
            elif topo_percentage > 0.2:
                print("⚠️  PARTIAL: Some topological learning but inconsistent")
            else:
                print("❌ POOR: Very limited topological learning")
            
            print(f"Final topological loss: {diagnosis['current_loss']:.4f}")
            print(f"Epochs with topology: {diagnosis['epochs_with_topology']}/{diagnosis['total_epochs']}")
        
        # Check if we avoided "three balls" problem
        clustering_results = evaluation_results.get('clustering', {})
        if 'clustering_accuracy' in clustering_results:
            clustering_acc = clustering_results['clustering_accuracy']
            if clustering_acc > 0.9:
                print(f"🎯 Excellent clustering accuracy: {clustering_acc:.3f}")
            elif clustering_acc > 0.7:
                print(f"👍 Good clustering accuracy: {clustering_acc:.3f}")
            else:
                print(f"⚠️  Poor clustering accuracy: {clustering_acc:.3f}")
        
        # Save final analysis
        final_analysis = {
            'experiment_config': config,
            'training_diagnosis': diagnosis,
            'evaluation_results': evaluation_results,
            'experiment_metadata': {
                'completion_time': datetime.now().isoformat(),
                'experiment_directory': exp_dir,
                'approach': 'contrastive_primary_topological_regularization',
                'insight': 'Pure topological loss fails because targets equal inputs'
            }
        }
        
        analysis_path = os.path.join(results_dir, 'final_analysis.json')
        with open(analysis_path, 'w') as f:
            # Convert any non-serializable objects
            import json
            serializable_analysis = {}
            for key, value in final_analysis.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_analysis[key] = value
                except:
                    serializable_analysis[key] = str(value)  # Convert to string if not
            
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"\nFinal analysis saved to: {analysis_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Experiment saved to: {exp_dir}")


if __name__ == "__main__":
    main_topological_training()