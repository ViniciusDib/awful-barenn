#!/bin/bash
#SBATCH --job-name=barenn_baseline
#SBATCH --partition=pluton_cpu
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=baseline_%j.out

echo "=== BASELINE LEGADO ==="
echo "No: $(hostname) | Fila: $SLURM_JOB_PARTITION"
lscpu | grep -E "Model name|MHz|Socket|Core"
echo ""

g++ -O2 -std=c++17 -o barenn barenn.cpp
echo "--- Compilado com -O2 -std=c++17 ---"
echo ""

echo "--- 5 execucoes com hyperfine ---"
hyperfine --warmup 1 --runs 5 './barenn'

echo ""
echo "--- Callgrind (1 execucao) ---"
valgrind --tool=callgrind --callgrind-out-file=callgrind_baseline.out ./barenn 2>&1 | tail -5
echo "Arquivo callgrind: callgrind_baseline.out"

echo ""
echo "--- Perf stat ---"
perf stat -e cache-misses,cache-references,instructions,cycles,branches,branch-misses ./barenn 2>&1
