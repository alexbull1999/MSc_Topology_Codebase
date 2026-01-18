#!/bin/bash
#SBATCH --job-name=xnli_tokens
#SBATCH --partition=a16
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=../logs/XNLI_TOKEN_EXTRACTION_%j.out
#SBATCH --error=../logs/XNLI_TOKEN_EXTRACTION_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting XNLI SBERT Token Extraction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load CUDA
. /vol/cuda/12.0.0/setup.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check GPU
nvidia-smi

cd ~/MSc_Topology_Codebase

export PYTHONUNBUFFERED=1

# Extract tokens for all languages
python phd_method/src_phd/xnli_sbert_token_extractor.py --language all

EXIT_CODE=$?

echo "Time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== TOKEN EXTRACTION SUCCESSFUL ==="
else
    echo "=== TOKEN EXTRACTION FAILED ==="
fi

exit $EXIT_CODE