#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -J analyze_L300
#SBATCH -o logs/analyze_L300_%j.out
#SBATCH -e logs/analyze_L300_%j.err

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb

echo "Starting analysis for L=300 Gibbs Real chains..."
$PYTHON scripts/analyze_chains.py --lmax 300 --nside 256 --sampler gibbs
echo "Analysis complete."
