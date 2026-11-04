#!/usr/bin/env python3
# Independent check of the file written by check_repro_nn: recomputes every Adam rounding with
# exact rationals, compares Adam with torch.optim, and MaxPool with torch's max_pool2d.
# Usage: python3 check_repro_nn.py <file>     (needs torch)
import math
import struct
import sys
from fractions import Fraction

import torch


def f(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rn(v):
    """Exact rational v rounded to the nearest float32, ties to even."""
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
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2):
        n += 1
    return s * float(Fraction(n) * Fraction(2) ** lsb)


def F(x):
    return Fraction(x)


def add(a, b):
    return rn(F(a) + F(b))


def mul(a, b):
    return rn(F(a) * F(b))


def div(a, b):
    return rn(F(a) / F(b))


def sqrt_rn(a):
    """float32 sqrt through a correctly rounded double sqrt is exact (53 >= 2 * 24 + 2 bits)."""
    return 0.0 if a == 0 else rn(F(math.sqrt(a)))


def main():
    lines = open(sys.argv[1]).read().splitlines()
    i = bad = pools = adams = 0
    worst_pool = worst_adam = 0.0
    while i < len(lines):
        head = lines[i].split()
        if head[0] == "pool":
            N, C, H, W, KH, KW, SY, SX = map(int, head[1:])
            x = [f(int(v, 16)) for v in lines[i + 1].split()]
            o = [f(int(v, 16)) for v in lines[i + 2].split()]
            ids = [int(v) for v in lines[i + 3].split()]
            dy = [f(int(v, 16)) for v in lines[i + 4].split()]
            dx = [f(int(v, 16)) for v in lines[i + 5].split()]
            i += 6
            X = torch.tensor(x, dtype=torch.float32).reshape(N, C, H, W).requires_grad_(True)
            Y, I = torch.nn.functional.max_pool2d(X, (KH, KW), (SY, SX), return_indices=True)
            if not torch.equal(Y.detach().flatten(), torch.tensor(o, dtype=torch.float32)):
                bad += 1
                print("pool output differs from torch", head)
            if not torch.equal(I.flatten(), torch.tensor(ids)):
                bad += 1
                print("pool index differs from torch", head)
            X64 = X.detach().double().requires_grad_(True)
            Y64 = torch.nn.functional.max_pool2d(X64, (KH, KW), (SY, SX))
            Y64.backward(torch.tensor(dy, dtype=torch.float64).reshape(Y64.shape))
            d = (X64.grad.flatten() - torch.tensor(dx, dtype=torch.float64)).abs().max().item()
            worst_pool = max(worst_pool, d)
            pools += 1
        else:
            n, steps, lrb, wdb, adamw = int(head[1]), int(head[2]), int(head[3], 16), int(head[4], 16), int(head[5])
            lr, wd = f(lrb), f(wdb)
            p = [f(int(v, 16)) for v in lines[i + 1].split()]
            i += 2
            b1, b2, eps = (struct.unpack("f", struct.pack("f", c))[0] for c in (0.9, 0.999, 1e-8))
            m, v = [0.0] * n, [0.0] * n
            bp1, bp2 = 1.0, 1.0
            P = torch.tensor(p, dtype=torch.float64).requires_grad_(True)
            cls = torch.optim.AdamW if adamw else torch.optim.Adam
            opt = cls([P], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
            for _ in range(steps):
                g = [f(int(x, 16)) for x in lines[i].split()]
                want = [int(x, 16) for x in lines[i + 1].split()]
                i += 2
                bp1, bp2 = mul(bp1, b1), mul(bp2, b2)
                ob1, ob2 = add(1.0, -b1), add(1.0, -b2)
                sq = sqrt_rn(add(1.0, -bp2))
                neg_step = -div(lr, add(1.0, -bp1))
                dec = mul(lr, wd) if wd != 0 and adamw else 0.0
                for j in range(n):
                    gg = g[j]
                    if wd != 0 and not adamw:
                        gg = add(gg, mul(wd, p[j]))
                    if dec != 0:
                        p[j] = add(p[j], -mul(dec, p[j]))
                    m[j] = add(mul(m[j], b1), mul(gg, ob1))
                    v[j] = add(mul(v[j], b2), mul(mul(gg, gg), ob2))
                    denom = add(div(sqrt_rn(v[j]), sq), eps)
                    p[j] = add(p[j], mul(div(m[j], denom), neg_step))
                    if bits(p[j]) != want[j]:
                        bad += 1
                P.grad = torch.tensor(g, dtype=torch.float64)
                opt.step()
            got = torch.tensor([f(x) for x in want], dtype=torch.float64)
            worst_adam = max(worst_adam, (P.detach() - got).abs().max().item())
            adams += 1
    print("exact and torch comparison: %d pool cases, %d Adam runs, %d values wrong" % (pools, adams, bad))
    print("worst differences from torch in double: pool backward %.2e, Adam weights after 25 steps %.2e" % (worst_pool, worst_adam))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
