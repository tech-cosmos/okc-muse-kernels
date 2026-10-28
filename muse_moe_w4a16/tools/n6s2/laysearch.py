# n6s2: pipe-cost layout search for dq. Output targets P (bf16x2 subnormal e2m1 bits). Output op types and ALU-pipe ops:
#  AND (q&M): 1 | SHL a (IMAD.SHL + LOP3): 1 ALU + 1 IMAD | ROT r (SHF + LOP3): 2 | SHR b (SHF/IMAD.HI + LOP3) |
#  MULK ((q & C) * K) & M, K <= 3 bits, no two contributions on one bit (no carries): 2 ALU + 1 IMAD.
# A layout = the 4 outputs' source sets partition the 32 q bits, AND the code bits must map consistently: each 4-bit code
# (sign, e1, e0, m) lands on (15|31, 8|24, 7|23, 6|22) of one output: any bijection is fine since _POS is free in prepare.
# Question: is there a layout with ALU <= 6 (two cost-1 outputs)?
import itertools
P = [6, 7, 8, 15, 22, 23, 24, 31]
ALL = set(range(32))
def src_shift(s):  # q << s (s > 0) or q >> -s, no wrap: source of target t = t - s
    S = [t - s for t in P]
    return S if all(0 <= x < 32 for x in S) else None
def src_rot(r): return [(t - r) % 32 for t in P]
cands = []  # (name, alu, imad, sources)
cands.append(("AND", 1, 0, P))
for s in range(1, 32):
    S = src_shift(s)
    if S: cands.append((f"SHL{s}", 1, 1, S))
    S = src_shift(-s)
    if S: cands.append((f"SHR{s}", 1, 1, S))  # IMAD.HI form
    cands.append((f"ROT{s}", 2, 0, src_rot(s)))
cheap = [c for c in cands if c[1] == 1]
print("cost-1 output types:", [c[0] for c in cheap])
def mulk_ok(R):
    """Can a single ((q & C) * K) & M map source set R (8 bits) onto P with K <= 3 bits and no carries? Any bijection."""
    R = sorted(R)
    for nk in (1, 2, 3):
        for K in itertools.combinations(range(0, 32), nk):
            # each target t must receive exactly one source r with t - r in K
            pos = {}
            ok = True
            for r in R:
                for k in K:
                    p = r + k
                    if p < 32:
                        if p in pos: ok = False; break
                        pos[p] = r
                if not ok: break
            if not ok: continue
            hit = {p: pos[p] for p in P if p in pos}
            if len(hit) == 8 and len(set(hit.values())) == 8: return K
    return None
best = []
for c1, c2 in itertools.combinations(cheap, 2):
    if set(c1[3]) & set(c2[3]): continue
    R = ALL - set(c1[3]) - set(c2[3])
    # remaining 16 bits -> two outputs: ROT/SHL/SHR/AND or MULK
    for c3 in cands:
        if not set(c3[3]) <= R: continue
        R4 = R - set(c3[3])
        for c4 in cands:
            if set(c4[3]) == R4: best.append((c1[0], c2[0], c3[0], c4[0], c1[1] + c2[1] + c3[1] + c4[1]))
        K = mulk_ok(R4)
        if K: best.append((c1[0], c2[0], c3[0], f"MULK{K}", c1[1] + c2[1] + c3[1] + 2))
    for c3 in [None]:
        pass
print("layouts with two cost-1 outputs:", len(best))
for b in sorted(set(best), key=lambda t: t[-1])[:20]: print(b)
# also: two cheap + split remaining 16 into two MULKs
n2 = 0
for c1, c2 in itertools.combinations(cheap, 2):
    if set(c1[3]) & set(c2[3]): continue
    R = sorted(ALL - set(c1[3]) - set(c2[3]))
    for A in itertools.combinations(R, 8):
        if mulk_ok(A) and mulk_ok(set(R) - set(A)):
            print("2 cheap + 2 MULK:", c1[0], c2[0], A); n2 += 1; break
print("2 cheap + 2 MULK layouts:", n2)
