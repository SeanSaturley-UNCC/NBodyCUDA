#!/bin/bash
#SBATCH --job-name=nbody_benchmark
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=nbody_benchmark_%j.out
#SBATCH --error=nbody_benchmark_%j.err

# Parameters
PARTICLES=10000
DT=0.01
NSTEPS=10
PRINT_EVERY=10

echo "===== N-Body Benchmark Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# CPU run
echo "----- Running CPU Version -----"
start_cpu=$(($(date +%s%N)/1000000))
./nbody $PARTICLES $DT $NSTEPS $PRINT_EVERY > nbody_cpu.out
end_cpu=$(($(date +%s%N)/1000000))
cpu_time=$((end_cpu - start_cpu))
echo "CPU run completed in $cpu_time ms"
echo ""

# GPU run
echo "----- Running GPU Version -----"
nvidia-smi
start_gpu=$(($(date +%s%N)/1000000))
./nbody_cuda $PARTICLES $DT $NSTEPS $PRINT_EVERY > nbody_gpu.out
end_gpu=$(($(date +%s%N)/1000000))
gpu_time=$((end_gpu - start_gpu))
echo "GPU run completed in $gpu_time ms"
echo ""

# Summary
echo "===== Benchmark Finished ====="
echo "CPU Time: $cpu_time ms"
echo "GPU Time: $gpu_time ms"

if [ $gpu_time -ne 0 ]; then
    speedup=$(echo "scale=2; $cpu_time/$gpu_time" | bc)
    echo "Speedup: ${speedup}x"
else
    echo "Speedup: GPU time = 0 ms, cannot calculate"
fi
