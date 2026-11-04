// Generated from why_via_double: float inputs whose exp/log lies within 2 double-ulps
// of a float rounding midpoint, with the correctly rounded float result (from 300-bit mpmath).
struct HardCase { int is_log; unsigned bits; double dist_ulp64; float correct; };
static const HardCase kHardCases[] = {
    {0, 0xc16912cdu, 1.265, 0x1.fa6636p-22f},
    {1, 0x0dc8bba4u, 0.996, -0x1.137588p+6f},
    {1, 0x111c87f8u, 1.291, -0x1.010bbep+6f},
    {1, 0x1a8446cbu, 1.689, -0x1.9a148ap+5f},
    {1, 0x1f116ab8u, 0.227, -0x1.676a7cp+5f},
    {1, 0x2c4c24b7u, 1.255, -0x1.a90e2p+4f},
    {1, 0x38dcbe38u, 0.554, -0x1.251748p+3f},
    {1, 0x3bf86ef0u, 0.823, -0x1.38733cp+2f},
    {1, 0x3c413d3au, 0.308, -0x1.1c2b1ep+2f},
    {1, 0x41178febu, 0.185, 0x1.1fcbcep+1f},
    {1, 0x464d5b2bu, 1.729, 0x1.2f79e2p+3f},
    {1, 0x4665a9a6u, 0.730, 0x1.330e4ap+3f},
    {1, 0x4c5d65a5u, 0.044, 0x1.1e0696p+4f},
    {1, 0x4d604ebeu, 0.077, 0x1.346a58p+4f},
    {1, 0x5ee8984eu, 0.759, 0x1.5c9442p+5f},
    {1, 0x65d890d3u, 0.030, 0x1.a9a3f2p+5f},
    {1, 0x66a8c860u, 0.233, 0x1.b2bc8cp+5f},
    {1, 0x6f31a8ecu, 0.456, 0x1.08b512p+6f},
    {1, 0x79e7ec37u, 0.868, 0x1.43ff6ep+6f},
};
