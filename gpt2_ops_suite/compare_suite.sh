#!/usr/bin/env bash
# Compare gpt2_ops_suite result files: for every operation, are the input hashes
# identical (same inputs) and the output hashes identical (same results)?
# Usage: ./compare_suite.sh results/*/*.txt
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <result file> <result file> [...]"; exit 1; }
for f in "$@"; do echo "  [$f] $(head -n 1 "$f" | sed 's/^# //')"; done
awk -F'\t' '
  FNR == 1 { nf++; label[nf] = FILENAME }
  $1 == "OP" {
    op = $2; inh = substr($4, 4); outh = substr($5, 5)
    if (!(op in seen)) { seen[op] = 1; order[++n] = op }
    in_h[op, nf] = inh; out_h[op, nf] = outh
  }
  END {
    printf "\n%-34s %-10s %s\n", "operation", "inputs", "outputs (all files)"
    bad = 0
    for (i = 1; i <= n; i++) {
      op = order[i]; same_in = 1; same_out = 1; missing = 0
      for (f = 1; f <= nf; f++) {
        if (!((op, f) in out_h)) { missing++; continue }
        if (in_h[op, f] != in_h[op, 1]) same_in = 0
        if (out_h[op, f] != out_h[op, 1]) same_out = 0
      }
      verdict = same_out ? "IDENTICAL" : "DIFFERENT"
      if (!same_out || !same_in) bad++
      printf "%-34s %-10s %s%s\n", op, same_in ? "same" : "DIFFER", verdict, missing ? sprintf(" (missing in %d file(s))", missing) : ""
      if (!same_out) for (f = 1; f <= nf; f++) if ((op, f) in out_h) printf "      %s  %s\n", out_h[op, f], label[f]
    }
    printf "\n%s\n", bad ? sprintf("%d operation(s) differ", bad) : "ALL OPERATIONS IDENTICAL ACROSS ALL FILES"
  }' "$@"
