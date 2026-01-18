#!/bin/bash
#SBATCH --job-name=xnli_phdim_all
#SBATCH --partition=a16
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=../logs/XNLI_PHDIM_ALL_SEEDS_%j.out
#SBATCH --error=../logs/XNLI_PHDIM_ALL_SEEDS_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting XNLI PH-Dim Analysis for all languages and seeds..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load CUDA
. /vol/cuda/12.0.0/setup.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Python: $(which python)"

# Check GPU
nvidia-smi

cd ~/MSc_Topology_Codebase

export PYTHONUNBUFFERED=1

# Run PH-Dim analysis for all languages and all seeds
python entailment_surfaces/phdim_distance_metric_xnli_sbert.py --language all

EXIT_CODE=$?

echo "Time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== ALL ANALYSES SUCCESSFUL ==="
else
    echo "=== SOME ANALYSES FAILED ==="
fi

exit $EXIT_CODE
