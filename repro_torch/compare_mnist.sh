#!/usr/bin/env bash
# Compares the hashes and loss bits of MNIST runs: IDENTICAL or the first difference.
# Usage: ./compare_mnist.sh results/*/*repro*.txt
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <file> <file> ..."; exit 1; }
base=$1; shift
fail=0
for f in "$@"; do
  if diff <(grep -E "^hash|bits" "$base") <(grep -E "^hash|bits" "$f") >/dev/null; then
    echo "IDENTICAL  $f"
  else
    echo "DIFFERENT  $f"
    diff <(grep -E "^hash|bits" "$base") <(grep -E "^hash|bits" "$f") | head -n 4
    fail=1
  fi
done
echo "(all compared with $base)"
exit $fail
