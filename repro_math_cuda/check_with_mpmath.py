#!/usr/bin/env python3
# Independent check of the file written by check_repro_math: recomputes each exp/log in
# 300-bit arithmetic, rounds it once to float32, and compares the bits.
# Usage: python3 check_with_mpmath.py <file>     (needs mpmath)
import multiprocessing
import struct
import sys

from mpmath import ceil, exp, floor, isinf, ldexp, log, mp, mpf

mp.prec = 300


def to_float(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def round_float32(v):
    """Bits of v rounded to the nearest float32, ties to even, with subnormals and overflow."""
    sign = 0x80000000 if v < 0 else 0
    v = abs(v)
    if v == 0:
        return sign
    _, e = mp.frexp(v)                       # v in [2^(e-1), 2^e)
    lsb = max(e - 24, -149)
    q = ldexp(v, -lsb)
    n = int(floor(q))
    frac = q - n
    if frac > 0.5 or (frac == 0.5 and n % 2):
        n += 1
    value = n * 2.0 ** lsb if lsb < 900 else 0.0
    if n >= 2 ** 24 and lsb + 24 > 127:      # overflow
        return sign | 0x7F800000
    if lsb + len(bin(n)) - 2 > 128:
        return sign | 0x7F800000
    return sign | struct.unpack("<I", struct.pack("<f", value))[0]


def check(line):
    fn, xb, ob = line.split()
    xb, ob = int(xb, 16), int(ob, 16)
    x = to_float(xb)
    if x != x or x in (float("inf"), float("-inf")):
        return None
    if fn == "exp":
        want = round_float32(exp(mpf(x)))
    else:
        if x <= 0 or x == 1.0:
            return None
        want = round_float32(log(mpf(x)))
    return None if want == ob else (fn, xb, ob, want)


def main():
    lines = open(sys.argv[1]).read().splitlines()
    with multiprocessing.Pool() as pool:
        bad = [r for r in pool.imap_unordered(check, lines, chunksize=4096) if r]
    print("mpmath: %d inputs checked, %d wrong" % (len(lines), len(bad)))
    for fn, x, o, w in bad[:20]:
        print("  %s(bits %08x): got %08x, correct %08x" % (fn, x, o, w))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
