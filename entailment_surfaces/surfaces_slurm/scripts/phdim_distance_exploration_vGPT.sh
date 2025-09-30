#!/bin/bash
#SBATCH --job-name=surface_distance_analysis
#SBATCH --partition=gpgpuC
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=../logs/GPT_SNLI_SURFACE_ANALYSIS_SEED4_%j.out
#SBATCH --error=../logs/GPT_SNLI_SURFACE_ANALYSIS_SEED4_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting Surface Distance Metric Analysis job..."
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

echo ""

# Change to your project directory
cd ~/MSc_Topology_Codebase

echo ""
echo "Starting Surface Distance Metric Analysis..."
echo "Analysis parameters:"
echo ""

# Run surface distance analysis
# Start with 20k samples for faster initial analysis
python entailment_surfaces/phdim_distance_metric_optimized_vGPT.py 


# Capture exit code
EXIT_CODE=$?

echo ""
echo "Analysis completed with exit code: $EXIT_CODE"
echo "Time: $(date)"

# Show what files were created
echo ""
echo "Results created in results/step_1_1_analysis/:"
ls -la results/step_1_1_analysis/ 2>/dev/null || echo "No results directory found"

# Show disk usage
echo ""
echo "Disk usage:"
du -sh results/step_1_1_analysis/ 2>/dev/null || echo "No results to measure"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== ANALYSIS SUCCESSFUL ==="
    echo "Surface distance metric analysis completed successfully!"
    echo ""
    echo "Key outputs:"
    echo "  - JSON results: results/step_1_1_analysis/comprehensive_surface_analysis_*.json"
    echo "  - Plain report: results/step_1_1_analysis/simple_analysis_report_*.txt"
    echo "  - Individual space results: results/step_1_1_analysis/surface_analysis_*_*.json"
    echo ""
    echo "Next steps:"
    echo "  1. Review the simple_analysis_report_*.txt for plain results"
    echo "  2. Identify top-performing space+metric combinations"
    echo "  3. Consider re-running with --max_samples 167000 for full analysis"
    echo ""
    echo "To run full analysis on promising combinations:"
    echo "  python step_1_1_surface_analysis_corrected.py \\"
    echo "    --bert_data 'data/processed/snli_full_standard_BERT.pt' \\"
    echo "    --order_model '$ORDER_MODEL' \\"
    echo "    --max_samples 167000"
else
    echo ""
    echo "=== ANALYSIS FAILED ==="
    echo "Please check the error output above for debugging information."
    echo ""
    echo "Common issues to check:"
    echo "  - Missing dependencies (topology.py, order_embeddings_asymmetry.py)"
    echo "  - Memory issues (try reducing --max_samples)"
    echo "  - CUDA out of memory (check GPU usage with nvidia-smi)"
fi

echo ""
echo "Job finished."