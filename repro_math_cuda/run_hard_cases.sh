#!/usr/bin/env bash
# RepDL's exp/log-via-double on the 19 hardest float inputs, on this GPU and this CPU's C library.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build "results/$(hostname -s)"
nvcc -O3 -std=c++17 -arch=native hard_cases.cu -o build/hard_cases
./build/hard_cases | tee "results/$(hostname -s)/hard_cases.txt"
