#!/bin/bash
#SBATCH --job-name=NBodyCUDA          # Job name
#SBATCH --partition=GPU               # Use the GPU partition
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks-per-node=1           # Single task
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=00:30:00               # Max walltime
#SBATCH --output=nbody_cuda.out       # Stdout log
#SBATCH --error=nbody_cuda.err        # Stderr log

module load cuda/12.4                 # Load CUDA module

# Run the executable
srun ./nbody_cuda planet 86400 100 10
