"""s31: evict_last cache hint on every cp.async.bulk.prefetch.L2 after `namespace g10 {` (proj/layer kernels + the
layer's upf injection into moe code; attention's prefetches come earlier in the file and stay unchanged).
python pf_last.py <in> <out> [site indices, e.g. 0,1,2 (default all)]"""
import sys
s = open(sys.argv[1]).read()
i = s.index("namespace g10 {")
a = "cp.async.bulk.prefetch.L2.global [%0], %1;"
b = "{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }"
print("sites:", s[i:].count(a))
parts = s[i:].split(a)
sel = set(range(len(parts) - 1)) if len(sys.argv) < 4 else {int(x) for x in sys.argv[3].split(",")}
t = parts[0] + "".join((b if j in sel else a) + parts[j + 1] for j in range(len(parts) - 1))
open(sys.argv[2], "w").write(s[:i] + t)
