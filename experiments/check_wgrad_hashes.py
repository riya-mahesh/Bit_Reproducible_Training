#!/usr/bin/env python3
"""Checks that the chunked weight gradient left every result unchanged.

Compares the final-weight hash of every repro run under a results folder with the hash the
no-atomics code produced for the same seed and batch size (same on every GPU and CUDA version).

Usage: python3 experiments/check_wgrad_hashes.py results/wgrad_batch_split
"""
import glob
import json
import os
import sys

# Final weights (epoch 2) of the no-atomics runs, by (seed, batch size).
EXPECTED = {
    (0, 50): "2ecd0e5156ce", (0, 100): "6b8ce5843c58", (0, 200): "8abbccf6ec9d",
    (0, 500): "6e2ad4e4cf7e", (0, 1000): "1ab462c2a075",
    (1, 100): "c955a8b61852", (2, 100): "20eea7769625", (3, 100): "7ad32ff9c8b3",
    (4, 100): "f5347082020d",
}

root = sys.argv[1] if len(sys.argv) > 1 else "results/wgrad_batch_split"
files = sorted(glob.glob(os.path.join(root, "**", "mnist", "*.json"), recursive=True))
checked = bad = 0
for f in files:
    d = json.load(open(f))
    if d.get("mode") != "repro":
        continue
    key = (d["seed"], d["batch"])
    got = d["hashes"]["weights_epoch_2"][:12]
    want = EXPECTED.get(key)
    if want is None:
        print("no reference  seed %d batch %d  %s  %s" % (key + (got, f)))
        continue
    checked += 1
    if got != want:
        bad += 1
        print("MISMATCH      seed %d batch %d  got %s want %s  %s" % (key + (got, want, f)))
print("%d repro runs checked, %d mismatches" % (checked, bad))
if checked == 0:
    print("No repro runs found under", root)
sys.exit(1 if bad or checked == 0 else 0)
