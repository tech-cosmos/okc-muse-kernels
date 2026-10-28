"""n5s23 (lead 10:55Z): can a PRMT byte-LUT decode of E2M1 beat twm dq() (8 int + 4 HMUL2 per 8 weights)?
Checks the facts the hand-derived bound in NOTES-proj n5s23 rests on."""
import struct, itertools
MAG = [0, .5, 1, 1.5, 2, 3, 4, 6]
def bf16(v): return struct.unpack('<I', struct.pack('<f', v))[0] >> 16
best = None
for E in range(-120, 120):          # value * 2^E: does some bias make the bytes fit one 8-byte PRMT table?
    w = [bf16(m * 2.0 ** E) for m in MAG]
    lo = {x & 0xFF for x in w}; hi = {x >> 8 for x in w}
    his = hi | {h | 0x80 for h in hi if h}   # with sign: -v sets bit 7 of the hi byte (and -0)
    his |= {0x80}
    n, ns = len(lo) + len(hi), len(lo) + len(his)
    if best is None or ns < best[1]: best = (n, ns, E, sorted(lo), sorted(hi))
    if E == 0: print('E=0: lo', [hex(x) for x in sorted(lo)], 'hi', [hex(x) for x in sorted(hi)])
print('min distinct bytes over all biases: unsigned %d (fits 8 B = 2 regs), signed %d (> 8: sign cannot come from the table)' % best[:2])
# per-weight selector info: unsigned needs (lo_idx, hi_idx) = 2 nibbles of the PRMT selector per weight
pairs = {(bf16(m) & 0xFF, bf16(m) >> 8) for m in MAG}
print('distinct (lo, hi) pairs:', len(pairs), '-> recoded magnitude needs 3 bits + sign 1 = all 4 stored bits; selector needs 8 bits/weight')
# lower bound per 8 weights (4 bf16x2 outputs): 1 PRMT per output; each selector needs >= 1 op (nibble bit 3 must be 0 and
# 2 nibbles/weight must be spread from 4 stored bits); sign needs 1 LOP3 per output; window shifts for 3 of 4 outputs.
prmt, sel, sgn, shf = 4, 4, 4, 2
print('PRMT-LUT bound per 8 weights: %d int + 4 HMUL2 = %d  vs dq() 8 int + 4 HMUL2 = 12' % (prmt + sel + sgn + shf, prmt + sel + sgn + shf + 4))
# pre-scaled per-group LUT: a lane holds 4 weights per row per k16 = per scale group (mma A fragment rows g/g+8, k 2t..2t+9)
print('pre-scaled LUT: 8 entries x 2 B = 16 B = 4 regs (lo/hi tables split -> +1 interleave PRMT per output), built per 4 weights:'
      ' >= 3 ops per 4 weights to build (IADD exponent doubling + 2 HMUL2) + >= 2 PRMT per output -> >= 14 per 8 weights')
