#!/usr/bin/env bash
# Shows what this machine offers. Run on each CIMS server and on a burst node, and read the output
# to choose configurations for 01_make_env.sh. Changes nothing.
echo "=== host ==="; hostname; cat /etc/os-release 2>/dev/null | grep PRETTY
echo "=== GPUs and driver (the 'CUDA Version' here is the newest this driver supports) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv 2>/dev/null || echo "no nvidia-smi here"
nvidia-smi 2>/dev/null | grep "CUDA Version"
echo "=== CUDA modules ==="; module avail cuda 2>&1 | head -n 30
echo "=== Python ==="; for p in python3 python3.9 python3.10 python3.11 python3.12; do command -v $p >/dev/null && echo "$p: $($p --version 2>&1)"; done
echo "=== compilers ==="; g++ --version | head -n 1; command -v nvcc && nvcc --version | sed -n 2p
echo "=== container tools ==="; for t in singularity apptainer conda; do command -v $t || true; done
echo "=== disk ==="; quota -s 2>/dev/null | tail -n 3; df -h "$HOME" | tail -n 1
for d in /scratch /scratch/$USER /tmp; do [ -d "$d" ] && df -h "$d" | tail -n 1; done
echo "=== Slurm ==="; command -v sbatch && sinfo -s 2>/dev/null | head -n 20; sacctmgr -n show assoc user=$USER format=account%30,partition%20 2>/dev/null | head -n 10
