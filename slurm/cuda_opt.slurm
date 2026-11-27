#!/bin/bash
#SBATCH --job-name=barenn_cuda_opt
#SBATCH --partition=pluton_gpu
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=cuda_opt_%j.out

echo "=== CUDA AVANCADO ==="
echo "No: $(hostname) | Fila: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv
echo ""

nvcc -arch=sm_86 -O3 -std=c++17 -o barenn_cuda_opt barenn_cuda_opt.cu
echo "--- Compilado com nvcc -arch=sm_86 -O3 -std=c++17 ---"
echo ""

echo "--- 3 execucoes com hyperfine ---"
hyperfine --warmup 1 --runs 3 './barenn_cuda_opt'

echo ""
echo "--- Nsight Systems profile ---"
nsys profile --stats=true --output=nsys_cuda_opt ./barenn_cuda_opt 2>&1 | tail -40

echo ""
echo "--- Nsight Compute (forward kernel, 1 launch) ---"
ncu --target-processes all --launch-count 1 --set full ./barenn_cuda_opt 2>&1 | head -60

echo ""
echo "--- Comparacao: batch_size=32 ---"
nvcc -arch=sm_86 -O3 -std=c++17 -DBATCH_SIZE=32 -o barenn_cuda_opt_b32 barenn_cuda_opt.cu 2>/dev/null
if [ $? -eq 0 ]; then
    echo "batch_size=32:"
    ./barenn_cuda_opt_b32 2>&1 | grep -E "Tempo|Acuracia|batch"
fi
