#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -J analyze_L200
#SBATCH -o logs/analyze_L200_%j.out
#SBATCH -e logs/analyze_L200_%j.err

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb

echo "Starting analysis for L=200 Gibbs Real chains..."
$PYTHON scripts/analyze_chains.py --lmax 200 --nside 128 --sampler gibbs
echo "Analysis complete."
