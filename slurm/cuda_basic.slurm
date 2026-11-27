#!/bin/bash
#SBATCH --job-name=barenn_cuda
#SBATCH --partition=pluton_gpu
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=cuda_basic_%j.out

echo "=== CUDA BASICO ==="
echo "No: $(hostname) | Fila: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv
echo ""

nvcc -arch=sm_86 -O2 -std=c++17 -o barenn_cuda barenn.cu
echo "--- Compilado com nvcc -arch=sm_86 -O2 -std=c++17 ---"
echo ""

echo "--- 3 execucoes com hyperfine ---"
hyperfine --warmup 1 --runs 3 './barenn_cuda'

echo ""
echo "--- Nsight Systems profile ---"
nsys profile --stats=true --output=nsys_cuda_basic ./barenn_cuda 2>&1 | tail -40

echo ""
echo "--- Nsight Compute (forward kernel, 1 launch) ---"
ncu --target-processes all --launch-count 1 --set full ./barenn_cuda 2>&1 | head -60
