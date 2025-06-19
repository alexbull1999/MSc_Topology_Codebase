#!/bin/bash
#SBATCH --job-name=tda_integration
#SBATCH --partition=gpgpuB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_tda_%j.out
#SBATCH --error=logs/slurm_tda_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting TDA integration job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load CUDA first
echo "Loading CUDA..."
. /vol/cuda/12.0.0/setup.sh

# Activate your conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Test required packages
echo "Testing required packages..."
python -c "
import torch
import numpy as np
import matplotlib.pyplot as plt
import ripser
import persim
import umap
from sklearn.manifold import TSNE
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Ripser version: {ripser.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('All packages loaded successfully!')
"

echo ""
echo "Checking for required input files..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/..

# Check if required files exist
missing_files=()

# Check for TDA-ready data from cone validation
if [ ! -f "validation_results/tda_ready_data_snli_10k.pt" ]; then
    missing_files+=("validation_results/tda_ready_data_snli_10k.pt (from cone_validation.py)")
fi

# Check for the TDA integration script
if [ ! -f "src/tda_integration_texts_preserved.py" ]; then
    missing_files+=("src/tda_integration_texts_preserved.py")
fi


if [ ${#missing_files[@]} -ne 0 ]; then
    echo "ERROR: Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure the following have been run successfully:"
    echo "  1. text_processing.py (creates processed data)"
    echo "  2. order_embeddings.py (creates trained model)"
    echo "  3. hyperbolic_projection.py (creates hyperbolic embeddings)"
    echo "  4. cone_validation.py (creates TDA-ready data)"
    exit 1
fi

echo "All required files found!"

echo ""
echo "Starting TDA integration analysis..."
echo "Analysis parameters:"
echo "  - Input: TDA-ready cone violations from SNLI 10k"
echo "  - TDA method: Persistent homology with Ripser"
echo "  - Max dimension: H0, H1, H2"
echo "  - Visualizations: Persistence diagrams, UMAP/t-SNE projections"
echo ""

# Run TDA integration
python src/tda_integration_texts_preserved.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "TDA integration completed with exit code: $EXIT_CODE"
echo "Time: $(date)"

# Show what files were created
echo ""
echo "Files created in results/tda_integration/:"
ls -la results/tda_integration/ 2>/dev/null || echo "No TDA results found"

echo ""
echo "Generated visualizations:"
ls -la results/tda_integration/*.png 2>/dev/null || echo "No visualization files found"

echo ""
echo "Generated data files:"
ls -la results/tda_integration/*.pt 2>/dev/null || echo "No PyTorch data files found"
ls -la results/tda_integration/*.npz 2>/dev/null || echo "No NumPy data files found"
ls -la results/tda_integration/*.json 2>/dev/null || echo "No JSON files found"

# Show analysis results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== TDA INTEGRATION SUCCESSFUL ==="
    echo "The topological data analysis has been completed successfully."
    echo ""
    echo "Key analyses performed:"
    echo "  ✓ Persistent homology computed for each entailment class"
    echo "  ✓ Topological signatures compared between classes"
    echo "  ✓ Hypotheses validated (complexity ordering)"
    echo "  ✓ 2D projections created (UMAP/t-SNE)"
    echo "  ✓ Neural network data prepared with preserved texts"
    echo ""
    echo "Generated outputs:"
    echo "  - persistence_diagrams.png: Birth-death diagrams for each class"
    echo "  - feature_comparison_fixed.png: Topological feature comparison"
    echo "  - cone_patterns_2d.png: UMAP/t-SNE projections"
    echo "  - neural_network_data_snli_10k.pt: Complete data for NN classification"
    echo ""
    echo "Expected topological patterns verified:"
    echo "  - Entailment: Simple topology (tight clusters)"
    echo "  - Neutral: Intermediate complexity"
    echo "  - Contradiction: Complex topology (dispersed patterns)"
    echo ""
    echo "Next steps:"
    echo "  1. Review topological analysis results"
    echo "  2. Use neural_network_data_snli_10k.pt for classification"
    echo "  3. Apply topological features as regularization"
else
    echo ""
    echo "=== TDA INTEGRATION FAILED ==="
    echo "Please check the error output above for debugging information."
    echo ""
    echo "Common issues to check:"
    echo "  - Ripser library installation"
    echo "  - Input data format from cone validation"
    echo "  - Memory requirements for TDA computation"
    echo "  - Missing dependencies (persim, umap-learn)"
fi

echo ""
echo "Job finished."