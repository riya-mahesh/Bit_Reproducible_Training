#!/usr/bin/env bash
# CPU proof of repro_expf and repro_logf on all 2^32 inputs, then an independent
# 300-bit recomputation of every slow-path input and a sample with mpmath.
#   ./run_check_math.sh
#   CORE_MATH_DIR=~/core-math ./run_check_math.sh     also compare against CORE-MATH
#       (git clone https://gitlab.inria.fr/core-math/core-math.git)
# Also writes results/<host>/cpu_repro.txt, comparable with GPU results via compare_math.sh.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results/$(hostname -s)${RESULT_TAG:+_$RESULT_TAG}; mkdir -p "$OUT" build
CXX=${CXX:-g++}; CC=${CC:-gcc}
FLAGS=(-O2 -std=c++17 -pthread -fno-fast-math -ffp-contract=off)
EXTRA=()
if [ -n "${CORE_MATH_DIR:-}" ]; then
  "$CC" -O2 -ffp-contract=off -c "$CORE_MATH_DIR/src/binary32/exp/expf.c" -o build/cm_expf.o
  "$CC" -O2 -ffp-contract=off -c "$CORE_MATH_DIR/src/binary32/log/logf.c" -o build/cm_logf.o
  EXTRA=(-DWITH_CORE_MATH build/cm_expf.o build/cm_logf.o)
fi
"$CXX" "${FLAGS[@]}" check_repro_math.cpp "${EXTRA[@]}" -o build/check_repro_math
./build/check_repro_math build/check_dump.txt | tee "$OUT/check_math.log"
python3 check_with_mpmath.py build/check_dump.txt | tee -a "$OUT/check_math.log"
"$CXX" "${FLAGS[@]}" exhaustive_math_cpu.cpp -o build/exhaustive_math_cpu
./build/exhaustive_math_cpu "$OUT/cpu_repro.txt" repro_expf repro_logf | tee -a "$OUT/check_math.log"
"$CXX" "${FLAGS[@]}" -I../repro_sum_cuda check_cross_entropy.cpp -o build/check_cross_entropy
./build/check_cross_entropy build/ce_cases.txt | tee -a "$OUT/check_math.log"
python3 check_cross_entropy.py build/ce_cases.txt | tee -a "$OUT/check_math.log"
