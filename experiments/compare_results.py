#!/usr/bin/env python3
"""Turns everything under results/ into the three answers:

  1. Are the hashes identical across GPUs, CUDA versions, and the CPU? (kernels and whole training)
  2. Training and test accuracy: ours, RepDL, stock PyTorch.
  3. Speed: ours, RepDL, stock PyTorch, on the same GPU and configuration.

Usage: python3 experiments/compare_results.py [--out report.md]    from the repository root
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def table(headers, rows):
    say("| " + " | ".join(headers) + " |")
    say("|" + "---|" * len(headers))
    for r in rows:
        say("| " + " | ".join(str(c) for c in r) + " |")
    say()


def short(h):
    return h[:12]


def load_mnist():
    runs = []
    for path in sorted(glob.glob("repro_torch/results/**/*.json", recursive=True) + glob.glob("results/mnist/**/*.json", recursive=True)):
        with open(path) as f:
            r = json.load(f)
        r["path"] = path
        runs.append(r)
    return runs


def config(r):
    tag = r.get("tag") or "untagged"
    return "%s / %s / torch %s / wheel CUDA %s / driver %s" % (r["device"], tag, r["torch"], r["cuda"], r["driver"])


def agree(groups):
    """groups: name -> {configuration -> value}. Returns rows and whether every name agrees."""
    ok = True
    rows = []
    for name in sorted(groups):
        values = groups[name]
        distinct = set(values.values())
        ok = ok and len(distinct) == 1
        rows.append((name, len(values), "yes" if len(distinct) == 1 else "**NO: %d different**" % len(distinct)))
    return rows, ok


def correctness(runs):
    say("# 1. Bit-reproducibility")
    say()
    all_ok = True
    all_complete = True

    # Whole training, per method and seed. Profiled runs change only timing, so they count too.
    say("## MNIST training")
    say()
    groups = defaultdict(lambda: defaultdict(set))
    for r in runs:
        if r.get("steps") == 1200:
            groups[(r["mode"], r["seed"], r["slices"])][config(r)].add(tuple(sorted(r["hashes"].items())))
    rows = []
    for (mode, seed, s), by_config in sorted(groups.items()):
        distinct = set().union(*by_config.values())
        weights = {dict(d)["weights_epoch_%d" % 2] for d in distinct if "weights_epoch_2" in dict(d)}
        same = len(distinct) == 1
        if mode == "repro" and not same:
            all_ok = False
        rows.append((mode, seed, s, len(by_config), "yes" if same else "**NO: %d different**" % len(distinct),
                     ", ".join(sorted(short(w) for w in weights))))
    if rows:
        table(["Method", "Seed", "S", "Configurations", "All hashes identical", "Final weights"], rows)
        say("Configurations compared:")
        say()
        for c in sorted({config(r) for r in runs}):
            say("- " + c)
        say()
    else:
        say("No complete training runs found.")
        say()

    # Whole-range exp and log.
    say("## exp and log over all 2^32 inputs")
    say()
    groups = defaultdict(dict)
    for path in glob.glob("repro_math_cuda/results/*/*repro*.txt"):
        with open(path) as f:
            for line in f:
                p = line.split()
                if p and p[0] == "SUMMARY" and not p[1].endswith("~nan"):
                    groups[p[1]][os.path.relpath(path, "repro_math_cuda/results")] = p[2]
    rows, ok = agree(groups)
    all_ok = all_ok and ok
    table(["Function", "Result files", "Identical"], rows) if rows else say("No results found.\n")

    # Every operation in the suite.
    say("## Kernels: every suite operation")
    say()
    groups = defaultdict(dict)
    paths = glob.glob("gpt2_ops_suite/results/*/*.txt")
    for path in paths:
        with open(path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p[0] == "OP":
                    groups[p[1]][os.path.relpath(path, "gpt2_ops_suite/results")] = p[4]
    # A run that crashed can leave an empty results file; its finished operations are still in suite.log.
    names = sorted(groups, key=len, reverse=True)
    for path in paths:
        rel = os.path.relpath(path, "gpt2_ops_suite/results")
        if any(rel in g for g in groups.values()):
            continue
        gpu = None
        with open(os.path.join(os.path.dirname(path), "suite.log"), errors="replace") as f:
            for line in f:
                if line.startswith("=== "):
                    gpu = line.split()[2].rstrip(":") if line.startswith("=== GPU") else None
                elif gpu is not None and os.path.basename(path).startswith("gpu" + gpu + "_"):
                    name = next((n for n in names if line.startswith(n + " ")), None)
                    hashes = re.findall(r"\b[0-9a-f]{16}\b", line)
                    if name and len(hashes) >= 2:
                        groups[name][rel] = "out=" + hashes[1]
    rows, ok = agree(groups)
    all_ok = all_ok and ok
    if rows:
        files = sorted(os.path.relpath(p, "gpt2_ops_suite/results") for p in paths)
        gpu_files = [f for f in files if "/gpu" in f]
        say("Of the result files, %d are from GPUs and %d from CPUs." % (len(gpu_files), len(files) - len(gpu_files)))
        say()
        gpu_ops = sum(1 for g in groups.values() if any("/gpu" in f for f in g))
        say("%d of %d operations have a result from at least one GPU." % (gpu_ops, len(groups)))
        say()
        different = [r for r in rows if not r[2] == "yes"]
        say("%d operations across %d result files: %s." % (len(rows), len(files), "all identical" if not different else "%d differ" % len(different)))
        say()
        if different:
            table(["Operation", "Result files", "Identical"], different)
        incomplete = [(name, len(groups[name]), len(files)) for name in sorted(groups) if len(groups[name]) < len(files)]
        if incomplete:
            all_complete = False
            say("**Incomplete: %d operations are missing from some result files, so they were not compared there.**" % len(incomplete))
            say()
            table(["Operation", "Files with a result", "Result files"], incomplete)
    else:
        say("No results found.")
        say()

    # Pass or fail lines from the test programs.
    say("## Test programs")
    say()
    rows = []
    for path in sorted(glob.glob("repro_*_cuda/results/*/*.log")):
        with open(path, errors="replace") as f:
            text = f.read()
        if "OVERALL" in text:
            passed = "OVERALL: FAIL" not in text
            all_ok = all_ok and passed
            rows.append((path, "pass" if passed else "**FAIL**"))
    table(["Log", "Result"], rows) if rows else say("No logs found.\n")
    say("**Our results so far: %s%s**" % ("no differences found" if all_ok else "DIFFERENCES OR FAILURES FOUND, see above",
                                         "" if all_complete else "; the suite comparison is INCOMPLETE, see above"))
    say()


def fmt(values, digits=4):
    if len(values) == 1:
        return "%.*f" % (digits, values[0])
    return "%.*f ± %.*f" % (digits, statistics.mean(values), digits, statistics.stdev(values))


def accuracy(runs):
    say("# 2. Accuracy after 2 epochs (mean ± standard deviation over seeds)")
    say()
    say("A method with identical results everywhere counts once per seed. Stock differs between runs, so every GPU run in a configuration with all the seeds counts.")
    say()
    complete = [r for r in runs if r.get("steps") == 1200 and not r.get("profile")]
    most_seeds = max((len({r["seed"] for r in complete if config(r) == config(x)}) for x in complete), default=0)
    rows = []
    for mode in ("repro", "repdl", "stock"):
        v = [r for r in complete if r["mode"] == mode]
        if mode == "stock":
            v = [r for r in v if r["device"] != "arm" and len({x["seed"] for x in v if config(x) == config(r)}) == most_seeds]
        else:
            v = list({r["seed"]: r for r in v}.values())
        if v:
            rows.append((mode, len({r["seed"] for r in v}), len(v), fmt([r["train_accuracy"] for r in v]), fmt([r["test_accuracy"] for r in v]),
                         fmt([r["final_loss"] for r in v], 3)))
    table(["Method", "Seeds", "Runs counted", "Train accuracy", "Test accuracy", "Last batch loss"], rows) if rows else say("No complete runs found.\n")


def speed(runs):
    say("# 3. Speed: median ms per training step")
    say()
    say("Same GPU, configuration, data, order, and initial weights. The ratio is against RepDL (above 1 means slower than RepDL).")
    say()
    by_config = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.get("steps") == 1200 and not r.get("profile"):
            by_config[config(r)][r["mode"]].append(r)
    rows = []
    for c in sorted(by_config):
        m = by_config[c]
        med = {k: statistics.median(x["step_ms_median"] for x in v) for k, v in m.items()}
        total = {k: statistics.median(x["train_seconds"] for x in v) for k, v in m.items()}
        row = [c]
        for k in ("repro", "repdl", "stock"):
            row.append("%.2f (%.1f s)" % (med[k], total[k]) if k in med else "-")
        row.append("%.2f" % (med["repro"] / med["repdl"]) if "repro" in med and "repdl" in med else "-")
        row.append("%.2f" % (med["repro"] / med["stock"]) if "repro" in med and "stock" in med else "-")
        for k in ("repro", "repdl", "stock"):
            row.append("%.0f" % statistics.median(x["peak_memory_mib"] for x in m[k]) if k in m else "-")
        rows.append(row)
    table(["Configuration", "Ours", "RepDL", "Stock", "Ours / RepDL", "Ours / stock", "Ours peak MiB", "RepDL peak MiB", "Stock peak MiB"], rows) if rows else say("No complete runs found.\n")
    prof = [r for r in runs if r.get("profile") and r.get("steps") == 1200]
    if prof:
        say("Where the time goes (median ms; each part ends with a synchronization):")
        say()
        table(["Configuration", "Method", "Forward", "Backward", "Optimizer"],
              [(config(r), r["mode"], "%.2f" % r["breakdown_ms"].get("forward", 0), "%.2f" % r["breakdown_ms"].get("backward", 0),
                "%.2f" % r["breakdown_ms"].get("optimizer", 0)) for r in sorted(prof, key=lambda x: (config(x), x["mode"]))])
    official = sorted(glob.glob("results/mnist/**/*repdl_official.log", recursive=True))
    if official:
        say("RepDL's own example (its README publishes trained-model hash 31ee86a7f75d...):")
        say()
        rows = []
        for path in official:
            with open(path) as f:
                text = f.read()
            start = re.search(r"initial model: ([0-9a-f]{12})", text)
            rows.append((path, start.group(1) if start else "-",
                         "matches" if "31ee86a7f75dbd76bac22f209617eb7349f93b0ede116dba924f328cac3013f1" in text else "**differs**"))
        table(["Log", "Starting weights", "Published trained hash"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not os.path.isdir("repro_torch"):
        sys.exit("run this from the repository root")
    runs = load_mnist()
    correctness(runs)
    accuracy(runs)
    speed(runs)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
