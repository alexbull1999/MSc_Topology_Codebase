#!/bin/bash
#SBATCH --job-name=xnli_sbert_all_langs
#SBATCH --partition=a16
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=../logs/xnli_processing_sbert_all_langs_%j.out
#SBATCH --error=../logs/xnli_processing_sbert_all_langs_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting XNLI SBERT Processing for All Languages..."
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

# Test PyTorch and CUDA
echo "Testing PyTorch and CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('PyTorch setup verified!')
"

# Change to your project directory
cd ~

echo ""
echo "=========================================="
echo "Processing All 15 XNLI Languages"
echo "=========================================="
echo ""

# Define all XNLI languages
languages=("en" "ar" "bg" "de" "el" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh")

# Counter for tracking progress
total_langs=${#languages[@]}
current=0
successful=0
failed=0

# Process each language
for lang in "${languages[@]}"
do
    current=$((current + 1))
    echo ""
    echo "=========================================="
    echo "Processing language $current/$total_langs: $lang"
    echo "=========================================="
    echo "Start time: $(date)"
    echo ""
    
    python MSc_Topology_Codebase/entailment_surfaces/xnli_text_processing_sbert.py --language $lang
    
    # Check exit code
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully processed $lang"
        successful=$((successful + 1))
    else
        echo ""
        echo "✗ Failed to process $lang"
        failed=$((failed + 1))
    fi
    
    echo "End time: $(date)"
    echo ""
done

# Final summary
echo ""
echo "=========================================="
echo "PROCESSING COMPLETE - SUMMARY"
echo "=========================================="
echo "Total languages: $total_langs"
echo "Successful: $successful"
echo "Failed: $failed"
echo ""
echo "Processed languages:"
for lang in "${languages[@]}"
do
    output_file="/vol/bitbucket/ahb24/tda_entailment_new/xnli_data/processed/xnli_${lang}_combined_SBERT.pt"
    if [ -f "$output_file" ]; then
        file_size=$(du -h "$output_file" | cut -f1)
        echo "  ✓ $lang - $file_size"
    else
        echo "  ✗ $lang - NOT FOUND"
    fi
done

echo ""
echo "Job finished: $(date)"
echo ""

# Exit with error if any language failed
if [ $failed -gt 0 ]; then
    echo "=== SOME LANGUAGES FAILED ==="
    exit 1
else
    echo "=== ALL LANGUAGES PROCESSED SUCCESSFULLY ==="
    exit 0
fi