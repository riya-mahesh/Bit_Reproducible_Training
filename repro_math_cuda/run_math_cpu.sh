#!/usr/bin/env bash
# Build and run the CPU (C library) version of the exhaustive math test.
# Result: results/<host>/cpu_<model>.txt, comparable with GPU results via compare_math.sh.
set -euo pipefail
cd "$(dirname "$0")"
HOST=$(hostname -s); OUT=results/$HOST; mkdir -p "$OUT" build
# No fast-math, no automatic FMA contraction, no CPU-specific instruction sets:
# the program's own arithmetic is plain IEEE; only the C library's functions vary.
g++ -O2 -std=c++17 -pthread -fno-fast-math -ffp-contract=off exhaustive_math_cpu.cpp -o build/exhaustive_math_cpu
model=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//; s/[^A-Za-z0-9]/_/g' || echo cpu)
echo "CPU: $(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2)"
echo "FMA support: $(grep -qw fma /proc/cpuinfo && echo yes || echo no)   (glibc picks different code paths for CPUs with FMA)"
./build/exhaustive_math_cpu "$OUT/cpu_${model}.txt" | tee "$OUT/math_cpu.log"
