#!/bin/bash
#SBATCH --job-name=test_text_processing
#SBATCH --partition=gpgpuB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_text_processing_%j.out
#SBATCH --error=logs/slurm_text_processing_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting text processing test job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load any required modules (adjust based on your setup)

# Load CUDA first
echo "Loading CUDA..."
. /vol/cuda/12.0.0/setup.sh

# Activate your conda environment - FIXED VERSION
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh


conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Conda info:"
conda info --envs

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Test Python packages
echo "Testing Python packages..."
timeout 60 python -c "
import torch
import transformers
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Transformers version: {transformers.__version__}')
print('All packages loaded successfully!')
"

# Capture the exit code immediately
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 124 ]; then
    echo "Python package test timed out after 60 seconds"
    exit 1
elif [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Python package test failed with exit code: $PYTHON_EXIT_CODE"
    exit 1
else
    echo "Python package test succeeded!"
fi

echo ""
echo "Running text_processing.py..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/..

# Run text processing
python src/text_processing.py

echo ""
echo "Text processing completed!"
echo "Time: $(date)"

# Show what files were created
echo "Files created in data/processed:"
ls -la data/processed/ 2>/dev/null || echo "No data/processed directory found"

echo "Job finished."