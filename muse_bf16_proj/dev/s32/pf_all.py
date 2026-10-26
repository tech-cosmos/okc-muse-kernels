"""s32: evict_last cache hint on cp.async.bulk.prefetch.L2 sites anywhere in the layer's submission.py (CUDA asm text,
incl. escaped copies inside Python strings, and the Triton boundary PF). python pf_all.py <in> <out> [sites|all|list]"""
import sys
s = open(sys.argv[1]).read()
A = "cp.async.bulk.prefetch.L2.global [%0], %1;"
B = "{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }"
TA = "cp.async.bulk.prefetch.L2.global [$1], 1024; mov.u32 $0, 0;"
TB = "{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [$1], 1024, pl; } mov.u32 $0, 0;"
pos = sorted([(i, A, B) for i in range(len(s)) if s.startswith(A, i)] + [(i, TA, TB) for i in range(len(s)) if s.startswith(TA, i)])
arg = sys.argv[3] if len(sys.argv) > 3 else "all"
if arg == "list":
    for j, (i, a, _) in enumerate(pos):
        print(j, s.count("\n", 0, i) + 1, s[s.rfind("\n", 0, i) + 1:i].strip()[:70])
    sys.exit()
sel = set(range(len(pos))) if arg == "all" else {int(x) for x in arg.split(",")}
out, last = [], 0
for j, (i, a, b) in enumerate(pos):
    out.append(s[last:i]); out.append(b if j in sel else a); last = i + len(a)
out.append(s[last:])
open(sys.argv[2], "w").write("".join(out))
print("patched", sorted(sel))
