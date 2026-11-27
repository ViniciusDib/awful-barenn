#!/bin/bash
#SBATCH --job-name=barenn_cpu_opt
#SBATCH --partition=pluton_cpu
#SBATCH --mem=2G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=cpu_opt_%j.out

echo "=== CPU OTIMIZADO ==="
echo "No: $(hostname) | Fila: $SLURM_JOB_PARTITION"
lscpu | grep -E "Model name|MHz|Socket|Core"
echo ""

g++ -O3 -fopenmp -march=native -std=c++17 -o barenn_cpu barenn_cpu.cpp
echo "--- Compilado com -O3 -fopenmp -march=native -std=c++17 ---"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "--- 5 execucoes com hyperfine (OMP_NUM_THREADS=$OMP_NUM_THREADS) ---"
hyperfine --warmup 1 --runs 5 "./barenn_cpu"

echo ""
echo "--- Scaling test: 1, 2, 4, 8, 16 threads ---"
for T in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$T
    echo "Threads=$T:"
    ./barenn_cpu 2>&1 | grep "Tempo"
done

echo ""
echo "--- Callgrind (1 thread, 1 execucao) ---"
export OMP_NUM_THREADS=1
valgrind --tool=callgrind --callgrind-out-file=callgrind_cpu_opt.out ./barenn_cpu 2>&1 | tail -5

echo ""
echo "--- Perf stat (todos os threads) ---"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
perf stat -e cache-misses,cache-references,instructions,cycles,branches,branch-misses ./barenn_cpu 2>&1
