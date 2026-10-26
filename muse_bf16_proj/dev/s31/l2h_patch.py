"""s31: L2 evict_first cache hints on proj kernels' weight streams inside the layer's submission.py.
python l2h_patch.py <in submission.py> <out> <ns,ns,...>   (ns from g10 g12 g13 g14 g19 pr1 pr8 r3 g17 n2u nu8)
ld.global.nc ... -> same with .L2::cache_hint + an evict_first policy (createpolicy, CSE-able); g19 W bulk copies
(ring stages) get .L2::cache_hint too (x copies unchanged). Outputs bitwise unchanged (hints only)."""
import re, sys
src, dst, nss = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
s = open(src).read()
POL = ('__device__ __forceinline__ unsigned long long l2ef() { unsigned long long p; '
       'asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }\n')
def region(ns):
    a = re.search(rf"namespace {ns} \{{\n", s).start()
    b = a + 5 + re.search(r"namespace \w+ \{", s[a + 5:]).start()
    return a, b
for ns in nss:
    a, b = region(ns)
    r = s[a:b]; n0 = r
    hdr = f"namespace {ns} {{\n"
    r = r.replace(hdr, hdr + POL, 1)
    r = r.replace('ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"\n                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));',
                  'ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;"\n                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(l2ef()));')
    r = r.replace('ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));',
                  'ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v2.u32 {%0,%1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(l2ef()));')
    r = r.replace('ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p));',
                  'ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(l2ef()));')
    if ns == "g19":
        bw = ('__device__ __forceinline__ void bulkw(unsigned dst, const void* src, int bytes, unsigned b) {\n'
              '    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n'
              '                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b), "l"(l2ef()) : "memory");\n}\n')
        i = r.index("__device__ __forceinline__ void bulk(")
        r = r[:i] + bw + r[i:]
        r = re.sub(r"\bbulk\(ring", "bulkw(ring", r)
    nld = r.count("l2ef()") - 1
    print(ns, "hinted sites:", nld)
    if nld == 0: print("  (no sites)")
    s = s[:a] + r + s[b:]
open(dst, "w").write(s)
