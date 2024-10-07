#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint="80gb"
#SBATCH --output=logs/job-%j.out

# Run the command inside the Python container, passing the third argument
apptainer run --nv python_container.sif bash -c ./exps/new_exps/rl_bigbench_gemma2.sh