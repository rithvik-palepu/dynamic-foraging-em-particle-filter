#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name mmlpf_cv_batch
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH -o mmlpf_cv_batch_%j.out

# Update the temporary path used by pip/conda 
export TMPDIR="/allen/programs/celltypes/workgroups/mousecelltypes/rithvik.palepu/"

# Activate your designated virtual environment
source activate torch19

# Navigate to the specific project directory where you cloned the repo
cd ~/dynamic-foraging-mml-particle-filter/mml-pf-dynamic-foraging/

# Execute the python script
python psytrack_walk_forward_cv_comparison.py