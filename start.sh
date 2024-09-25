#!/bin/bash
#SBATCH --job-name=$1  # Job name as the first argument
#SBATCH --partition=$2  # Partition as the second argument
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

# Run the command inside the Python container, passing the third argument
apptainer run --nv python_container.sif $3
