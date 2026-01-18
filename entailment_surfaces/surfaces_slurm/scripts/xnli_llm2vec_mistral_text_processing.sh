#!/bin/bash
#SBATCH --job-name=xnli_llm2vec_all
#SBATCH --partition=a100
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=../logs/XNLI_LLM2VEC_MISTRAL_TEXT_PROCESSING_ALL_LANGUAGES_%j.out
#SBATCH --error=../logs/XNLI_LLM2VEC_MISTRAL_TEXT_PROCESSING_ALL_LANGUAGES_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting XNLI LLM2Vec processing for all languages..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load CUDA
echo "Loading CUDA..."
. /vol/cuda/12.0.0/setup.sh

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"

# Check GPU
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Test PyTorch and CUDA
echo "Testing PyTorch and CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

cd ~/MSc_Topology_Codebase

echo ""
echo "=========================================="
echo "Processing all XNLI languages with LLM2Vec-Mistral"
echo "=========================================="
echo ""

export PYTHONUNBUFFERED=1

# Process all languages (handled in Python)
python entailment_surfaces/xnli_text_processing_llm2vec.py --language all

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== ALL LANGUAGES PROCESSED SUCCESSFULLY ==="
else
    echo ""
    echo "=== SOME LANGUAGES FAILED ==="
fi

exit $EXIT_CODE