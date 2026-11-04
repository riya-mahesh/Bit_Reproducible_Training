#!/usr/bin/env bash
# Compare two result files from run_math_test.sh (two GPUs, or two machines).
# For each function: IDENTICAL (same outputs on all 2^32 inputs) or DIFFERENT,
# and for differences, which input ranges differ. "<name>~nan" is the same function
# with every NaN output replaced by one canonical bit pattern: if <name> differs but
# <name>~nan is identical, the two only disagree on the bit pattern of NaN results.
# Usage: ./compare_math.sh results/cuda2/gpu0_*.txt results/cuda5/gpu0_*.txt
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <file1> <file2>"; exit 1; }
echo "A: $(head -n 1 "$1")"
echo "B: $(head -n 1 "$2")"
awk '
  FNR == 1 { file++ }
  $1 == "SUMMARY" { sum[file, $2] = $3; if (file == 1) order[++n] = $2 }
  $1 == "B"       { blk[file, $2, $3] = $4 }
  END {
    for (i = 1; i <= n; i++) {
      f = order[i]
      if (!((2, f) in sum)) { printf "%-16s missing in B\n", f; continue }
      if (sum[1, f] == sum[2, f]) { printf "%-16s IDENTICAL on all 2^32 inputs\n", f; continue }
      nd = 0; list = ""
      for (k = 0; k < 4096; k++) if (blk[1, f, k] != blk[2, f, k]) {
        nd++
        if (nd <= 8) {
          sign = (k >= 2048) ? "-" : "+"; e = int(k / 8) % 256
          if (e == 0) r = "subnormal/zero"; else if (e == 255) r = "inf/nan"
          else r = sprintf("|x| in [2^%d, 2^%d)", e - 127, e - 126)
          list = list sprintf("\n    block %4d  (%s, %s)", k, sign, r)
        }
      }
      printf "%-16s DIFFERENT in %d of 4096 blocks%s\n", f, nd, list
      if (nd > 8) printf "    ... and %d more\n", nd - 8
    }
    print "To see exact differing values: run  build/exhaustive_math --dump <func> <block>  on both and diff the outputs."
  }' "$1" "$2"
