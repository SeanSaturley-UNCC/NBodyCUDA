# N-Body CUDA Benchmark

This repository contains CPU and GPU implementations of the N-Body simulation, along with a benchmark script to compare performance.

## Contents

- nbody.cpp — CPU implementation of the N-Body simulation
- nbody_cuda.cu — GPU implementation using CUDA
- Makefile — Build instructions
- run_benchmark.sh — SLURM batch script to run CPU and GPU benchmarks
- run_nbody_cuda.sh — Local script to run GPU version without SLURM

## Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- C++ compiler (g++ or similar)
- SLURM workload manager (for run_benchmark.sh)

## Build Instructions

From the `sequential` directory, run:
make



This produces:

- `nbody` — CPU executable
- `nbody_cuda` — GPU executable

To clean previous builds:
make clean


## Running the Benchmark

### Using SLURM
sbatch run_benchmark.sh


The script will:

1. Run the CPU version and measure time
2. Run the GPU version and measure time
3. Output CPU/GPU times and speedup to `nbody_benchmark.out`

Quickly check results:
grep "Time|Speedup" nbody_benchmark.out


### Running GPU Version Locally
./run_nbody_cuda.sh


## Notes

- Benchmark parameters (number of particles, steps, time step) can be adjusted in `run_benchmark.sh`
- Output files (`*.out`, `*.err`) are ignored by Git
