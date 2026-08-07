#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name mmlpf_cv_batch
#SBATCH --array=0-9
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH -o mmlpf_cv_batch_%A_task_%a.out

# Update the temporary path used by pip/conda 
export TMPDIR="/allen/programs/celltypes/workgroups/mousecelltypes/rithvik.palepu/"

# Activate designated virtual environment
source ~/miniconda3/bin/activate cv_env

# Navigate to the specific project directory where you cloned the repo
cd ~/dynamic-foraging-mml-particle-filter/mml-pf-dynamic-foraging/

# Execute the python script, passing the array ID as an argument
python psytrack_walk_forward_cv_comparison.py --array_id $SLURM_ARRAY_TASK_ID