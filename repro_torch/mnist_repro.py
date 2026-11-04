#!/usr/bin/env python3
"""LeNet-5 on MNIST with RepDL's settings (batch 100, Adam lr 0.01, 2 epochs).

  --mode repro   our reproducible layers (default)
  --mode stock   ordinary PyTorch layers: same data, order, and initial weights, for comparison
  --mode repdl   RepDL's layers, if the repdl package is installed

Prints hashes, training and test accuracy, and timing, and writes them to --out and, with the same
name ending in .json, for experiments/compare_results.py. All modes use the same data, order, and
initial weights. --profile also times forward, backward, and optimizer, which needs a
synchronization between them, so use it for the breakdown and not for the total.
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

if os.environ.get("REPRO_BUILD"):      # a build for one GPU and CUDA version; see experiments/
    sys.path.insert(0, os.environ["REPRO_BUILD"])
import repro_torch as rt

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    _, dims = struct.unpack(">HH", data[:4])
    shape = struct.unpack(">" + "I" * (dims & 0xFF), data[4:4 + 4 * (dims & 0xFF)])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * (dims & 0xFF)).reshape(shape)


def load_mnist(root):
    os.makedirs(root, exist_ok=True)
    for name, md5 in FILES.items():
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(BASE + name, path)
        with open(path, "rb") as f:
            if hashlib.md5(f.read()).hexdigest() != md5:
                raise RuntimeError("%s does not match its known checksum; delete it and rerun" % path)
    # byte / 255 in float32: one IEEE division, and the same values as torchvision's ToTensor.
    def images(name):
        return (load_idx(os.path.join(root, name)).astype(np.float32) / np.float32(255.0))[:, None, :, :]

    def labels(name):
        return load_idx(os.path.join(root, name)).astype(np.int64)

    return (images("train-images-idx3-ubyte.gz"), labels("train-labels-idx1-ubyte.gz"),
            images("t10k-images-idx3-ubyte.gz"), labels("t10k-labels-idx1-ubyte.gz"))


def lenet(layers):
    Conv, Lin, Relu, Pool = layers
    return torch.nn.Sequential(
        Conv(1, 6, kernel_size=5, padding=2), Relu(), Pool(2, 2),
        Conv(6, 16, kernel_size=5), Relu(), Pool(2, 2),
        torch.nn.Flatten(),
        Lin(16 * 5 * 5, 120), Relu(),
        Lin(120, 84), Relu(),
        Lin(84, 10),
    )


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def driver_version():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "none"


def evaluate(model, x, y, batch):
    """Accuracy and the logits, in batches. On the CPU np.argmax takes the first of any tie, so
    accuracy has one defined value too."""
    outputs = []
    with torch.no_grad():
        for i in range(0, len(y), batch):
            outputs.append(model(x[i:i + batch]))
    logits = torch.cat(outputs)
    return float((np.argmax(logits.cpu().numpy(), axis=1) == y).mean()), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="repro", choices=["repro", "stock", "repdl"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--slices", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--max-steps", type=int, default=0, help="stop early, for a quick test")
    ap.add_argument("--profile", action="store_true", help="time forward, backward, and optimizer separately")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for every method; 0 is the default")
    ap.add_argument("--no-scan", action="store_true", help="skip the NaN and infinity scan, only to measure its cost")
    ap.add_argument("--tag", default=os.environ.get("RESULT_TAG", ""), help="configuration name in the results")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines = []

    def say(*a):
        text = " ".join(str(x) for x in a)
        print(text, flush=True)
        lines.append(text)

    if args.device.startswith("cuda") and not rt.has_cuda and args.mode == "repro":
        raise SystemExit("this build of repro_torch is CPU-only; rebuild with nvcc available")
    if args.threads:
        torch.set_num_threads(args.threads)
        os.environ["REPRO_CPU_THREADS"] = str(args.threads)
    if args.mode == "repdl":           # as in RepDL's own training example
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        name = torch.cuda.get_device_name(0)
    else:
        name = platform.processor() or platform.machine()
    say("mode", args.mode, "device", args.device, name, "torch", torch.__version__, "cuda", torch.version.cuda)
    say("batch", args.batch, "lr", args.lr, "epochs", args.epochs, "slices", args.slices, "seed", args.seed)

    xtr, ytr, xte, yte = load_mnist(args.data)
    say("hash data", hashlib.sha256(xtr.tobytes() + ytr.tobytes() + xte.tobytes() + yte.tobytes()).hexdigest())

    criterion = torch.nn.functional.cross_entropy
    if args.mode == "repro":
        rt.set_slices(args.slices)
        model = lenet((rt.Conv2d, rt.Linear, rt.ReLU, rt.MaxPool2d))
        rt.name_layers(model)
        criterion = rt.cross_entropy
        make_opt = lambda p: rt.Adam(p, lr=args.lr)
    elif args.mode == "stock":
        model = lenet((torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.MaxPool2d))
        make_opt = lambda p: torch.optim.Adam(p, lr=args.lr)
    else:
        import repdl
        model = lenet((torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.MaxPool2d))
        model = repdl.from_torch_module(model)
        criterion = repdl.nn.functional.cross_entropy
        make_opt = lambda p: repdl.optim.Adam(p, lr=args.lr)

    rt.reproducible_init(model, args.seed)
    init_hash = rt.model_hash(model)
    say("hash init", init_hash)
    device = args.device
    model.to(device)
    optimizer = make_opt(model.parameters())

    x_train, y_train = torch.from_numpy(xtr).to(device), torch.from_numpy(ytr).to(device)
    x_test, y_test = torch.from_numpy(xte).to(device), torch.from_numpy(yte).to(device)
    rng = rt.SplitMix64(args.seed + 1)
    guard = rt.layers.guard()
    if args.no_scan:
        guard.scan = lambda x, name: None
    times, parts, curve = [], {"forward": [], "backward": [], "optimizer": []}, []
    step = 0
    done = False
    hashes = {}
    last_loss = None
    for epoch in range(args.epochs):
        order = torch.tensor(rt.shuffled(len(xtr), rng), device=device)
        for i in range(0, len(xtr) - args.batch + 1, args.batch):
            sync(device)
            t0 = time.perf_counter()
            idx = order[i:i + args.batch]
            loss = criterion(model(x_train[idx]), y_train[idx])
            if args.profile:
                sync(device)
                t1 = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            if args.profile:
                sync(device)
                t2 = time.perf_counter()
            optimizer.step()
            if args.mode == "repro":
                guard.check(step)
            sync(device)
            t3 = time.perf_counter()
            times.append(t3 - t0)
            if args.profile:
                parts["forward"].append(t1 - t0)
                parts["backward"].append(t2 - t1)
                parts["optimizer"].append(t3 - t2)
            step += 1
            last_loss = loss
            if step % 100 == 0:
                value = loss.item()
                say("  epoch %d step %d loss %.4f (bits %08x)" % (epoch + 1, step, value,
                                                                  struct.unpack("<I", struct.pack("<f", value))[0]))
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        h = rt.model_hash(model)
        hashes["weights_epoch_%d" % (epoch + 1)] = h
        say("hash weights after epoch %d (%d steps) %s" % (epoch + 1, step, h))
        model.eval()
        test_accuracy, _ = evaluate(model, x_test, yte, args.batch)
        model.train()
        curve.append({"epoch": epoch + 1, "step": step, "test_accuracy": test_accuracy})
        say("test accuracy after epoch %d: %.4f" % (epoch + 1, test_accuracy))
        if done:
            break

    # Final accuracy on both sets; neither takes part in the training time.
    model.eval()
    test_accuracy, test_logits = evaluate(model, x_test, yte, args.batch)
    train_accuracy, train_logits = evaluate(model, x_train, ytr, args.batch)
    hashes["test_logits"] = rt.tensor_hash(test_logits)
    hashes["train_logits"] = rt.tensor_hash(train_logits)
    hashes["init"] = init_hash
    say("hash test logits", hashes["test_logits"])
    say("hash train logits", hashes["train_logits"])
    say("train accuracy %.4f" % train_accuracy)
    say("test accuracy %.4f" % test_accuracy)
    t = sorted(times[5:]) if len(times) > 10 else sorted(times)
    say("time train total %.2f s; per step median %.3f ms, mean %.3f ms (%d steps)" % (
        sum(times), 1e3 * statistics.median(t), 1e3 * statistics.mean(t), len(times)))
    peak = 0.0
    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 2.0 ** 20
        say("peak GPU memory %.1f MiB" % peak)
    breakdown = {}
    if args.profile:
        breakdown = {k: 1e3 * statistics.median(sorted(v[5:])) for k, v in parts.items() if len(v) > 5}
        say("median ms per step:", " ".join("%s %.3f" % kv for kv in breakdown.items()))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        record = {
            "mode": args.mode, "tag": args.tag, "host": platform.node(), "device": name, "driver": driver_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "seed": args.seed, "slices": args.slices,
            "batch": args.batch, "lr": args.lr, "epochs": args.epochs, "steps": len(times), "profile": args.profile,
            "hashes": hashes, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "curve": curve,
            "final_loss": last_loss.item(), "train_seconds": sum(times),
            "step_ms_median": 1e3 * statistics.median(t), "step_ms_mean": 1e3 * statistics.mean(t),
            "step_ms_p95": 1e3 * t[int(0.95 * (len(t) - 1))], "breakdown_ms": breakdown, "peak_memory_mib": peak,
        }
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
