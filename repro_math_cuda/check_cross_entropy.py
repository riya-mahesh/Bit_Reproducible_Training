#!/usr/bin/env python3
# Independent check of the file written by check_cross_entropy: recomputes each step of the
# definition exactly (exact rationals for sums and divisions, 300-bit exp and log), rounds each
# to float32, compares the bits, and compares with PyTorch's double-precision cross-entropy.
# Usage: python3 check_cross_entropy.py <file>     (needs mpmath; uses torch if installed)
import struct
import sys
from fractions import Fraction

from mpmath import exp, log, mp, mpf

mp.prec = 300


def to_f(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def to_b(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def rn(v):
    """Rational or mpf v rounded to the nearest float32, ties to even."""
    v = Fraction(int(v.as_integer_ratio()[0]), int(v.as_integer_ratio()[1])) if isinstance(v, float) else v
    if isinstance(v, mpf):
        m, e = mp.frexp(v)
        v = Fraction(int(mp.ldexp(m, 300))) * Fraction(2) ** (e - 300)
    if v == 0:
        return 0.0
    s = -1 if v < 0 else 1
    v = abs(v)
    e = v.numerator.bit_length() - v.denominator.bit_length()
    if Fraction(2) ** e > v:
        e -= 1
    lsb = max(e - 23, -149)
    q = v / Fraction(2) ** lsb
    n = q.numerator // q.denominator
    frac = q - n
    if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and n % 2):
        n += 1
    value = Fraction(n) * Fraction(2) ** lsb
    if value >= Fraction(2) ** 128:
        return s * float("inf")
    return s * float(value)


def ex(f):
    return Fraction(f)


def main():
    lines = open(sys.argv[1]).read().splitlines()
    i, cases, bad = 0, 0, 0
    try:
        import torch
    except ImportError:
        torch = None
    worst_loss = worst_grad = 0.0
    while i < len(lines):
        _, rows, cols, lb = lines[i].split()
        rows, cols, lb = int(rows), int(cols), int(lb, 16)
        t = [int(v) for v in lines[i + 1].split()]
        i += 2
        x, g = [], []
        for _ in range(rows * cols):
            a, b = lines[i].split()
            x.append(to_f(int(a, 16)))
            g.append(int(b, 16))
            i += 1
        count = sum(1 for v in t if v != -100)
        losses = Fraction(0)
        for r in range(rows):
            row = x[r * cols:(r + 1) * cols]
            m = max(row)
            e = [rn(exp(mpf(rn(ex(v) - ex(m))))) for v in row]
            S = rn(sum(ex(v) for v in e))
            if t[r] != -100:
                sh = rn(ex(row[t[r]]) - ex(m))
                losses += ex(rn(ex(rn(log(mpf(S)))) - ex(sh)))
            for c in range(cols):
                if t[r] == -100:
                    want = 0.0
                else:
                    p = rn(ex(e[c]) / ex(S))
                    if c == t[r]:
                        p = rn(ex(p) - 1)
                    want = rn(ex(p) / count)
                if to_b(want) != g[r * cols + c]:
                    bad += 1
        L = rn(losses / count)
        if to_b(L) != lb:
            bad += 1
            print("loss bits differ: got %08x, correct %08x" % (lb, to_b(L)))
        if torch is not None:
            X = torch.tensor(x, dtype=torch.float64).reshape(rows, cols).requires_grad_(True)
            T = torch.tensor(t)
            l = torch.nn.functional.cross_entropy(X, T, ignore_index=-100)
            l.backward()
            G = torch.tensor([to_f(v) for v in g], dtype=torch.float64).reshape(rows, cols)
            worst_loss = max(worst_loss, abs(l.item() - to_f(lb)) / max(abs(l.item()), 1e-30))
            worst_grad = max(worst_grad, (X.grad - G).abs().max().item() / max(X.grad.abs().max().item(), 1e-30))
        cases += 1
    print("exact definition: %d cases, %d values wrong" % (cases, bad))
    if torch is not None:
        print("vs PyTorch double: worst relative loss difference %.2e, worst gradient difference %.2e" % (worst_loss, worst_grad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
