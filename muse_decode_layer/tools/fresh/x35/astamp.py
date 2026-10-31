"""fresh x35: per-CTA %globaltimer stamps of the d1 / v1 SLIDING attention hops in-layer (timing copy only).
Patches submission.py's csl::decode (d1 sl), vtk::verify (v1 sl) and g13::gemv (d1 qkv K=8192 / o K=1024) to write
u64 stamps to a fixed device buffer (address baked into the patched source; recompiles those modules each run).
A stamp that must wait for a load reads the timer under a predicate computed from the loaded value (hard dependency).
python tools/fresh/x35/astamp.py <submission.py> ['decode 1 sliding'|'verify 1 sliding'] [reps]"""
import sys, os, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench

src, flt = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "decode 1 sliding"
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
NR, NC, NK = 8, 1024, 16  # regions x CTAs x slots (u64)
buf = torch.zeros(NR * NC * NK, dtype=torch.int64, device="cuda")
s = open(src).read()

HDR = ('\n#define STP_ADDR %dULL\n'
       '__device__ __forceinline__ unsigned long long gtd(unsigned d) { unsigned long long t; asm volatile("{\\n\\t.reg .pred p;\\n\\t'
       'setp.eq.u32 p, %%1, 0x7fc0dead;\\n\\t@p mov.u64 %%0, 0;\\n\\t@!p mov.u64 %%0, %%%%globaltimer;\\n\\t}" : "=l"(t) : "r"(d)); return t; }\n'
       '#define TSR(r, k, d) (reinterpret_cast<unsigned long long*>(STP_ADDR))[((r) * %d + blockIdx.x) * %d + (k)] = gtd(d)\n') % (buf.data_ptr(), NC, NK)


def rep(a, b, start=0, end=None):
    global s
    i = s.find(a, start, end if end is not None else len(s))
    assert i >= 0, a[:80]
    s = s[:i] + b + s[i + len(a):]
    return i


# --- csl / vtk (both in _CSL_SRC) ---
i = s.find("namespace csl {")
rep("namespace csl {", HDR + "namespace csl {", i)
for ns, reg, ql in (("namespace csl {", 2, "CSL_QLOADS"), ("namespace vtk {", 3, "VTK_QLOADS")):
    a = s.find(ns)
    e = s.find("}  // namespace " + ns.split()[1], a)
    R = str(reg)
    rep("    asm volatile(\"barrier.cluster.arrive.relaxed.aligned;\" ::: \"memory\");\n",
        "    asm volatile(\"barrier.cluster.arrive.relaxed.aligned;\" ::: \"memory\");\n    if (tid == 0) TSR(" + R + ", 0, 1u);\n", a, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    # after the page-table-dependent gathers are issued: pt arrival (slot 1), q arrival when !PW (slot 8)
    j = s.find("#undef CSL_LOADS", a, e)
    rep("#undef CSL_LOADS\n", "#undef CSL_LOADS\n    if (tid == 0) TSR(" + R + ", 1, (unsigned)kslot[0] ^ (unsigned)vslot[7]);\n"
        "    if (!PW && tid == 0) TSR(" + R + ", 8, qa.x ^ qb.w);\n", j, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    rep("        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n        " + ql + "\n",
        "        if (tid == 0) TSR(" + R + ", 9, 1u);\n        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n        if (tid == 0) TSR(" + R + ", 10, 1u);\n        " + ql + "\n"
        "        if (tid == 0) TSR(" + R + ", 8, qa.x ^ qb.w);\n", a, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    rep("    float sc[4][4];\n", "    if (tid == 0) TSR(" + R + ", 2, kr[0].x ^ kr[3].w ^ vr[0].x ^ vr[7].y);\n    float sc[4][4];\n", a, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    k = s.find("        float* lp = &lpart", a, e); k = s.rfind("    {\n", a, k)
    rep("    {\n", "    if (tid == 0) TSR(" + R + ", 3, __float_as_uint(acc[0][0]) ^ __float_as_uint(acc[3][3]));\n    {\n", k, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    rep("    __syncthreads();\n    {\n        const int h2", "    __syncthreads();\n    if (tid == 0) TSR(" + R + ", 4, 1u);\n    {\n        const int h2", a, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    rep("        const unsigned owner = row;\n", "        if (tid == 0) TSR(" + R + ", 5, 1u);\n        const unsigned owner = row;\n", a, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    rep("::\"r\"(mb) : \"memory\");\n        float M = -1e30f;", "::\"r\"(mb) : \"memory\");\n        if (tid == 0) TSR(" + R + ", 6, 1u);\n        float M = -1e30f;", a, e)
    e = s.find("}  // namespace " + ns.split()[1], a)
    k = s.find("__floats2bfloat162_rn(o0, o1);\n", a, e)
    rep("__floats2bfloat162_rn(o0, o1);\n", "__floats2bfloat162_rn(o0, o1);\n        if (tid == 0) { __threadfence(); TSR(" + R + ", 7, 1u); }\n", k, e)

# --- g13 (d1 qkv K 8192 -> region 0, o K 1024 -> region 1) ---
a = s.find("_G13_SRC = r'''")
rep("namespace g13 {", HDR + "namespace g13 {", a)
a = s.find("namespace g13 {", a)
e = s.find("}  // namespace g13", a)
rep("    const int nch = n / RPC;\n", "    const int nch = n / RPC;\n    if (threadIdx.x == 0) TSR((KK == 8192 ? (n == 2304 ? 0 : 4) : 1), 0, 1u);\n", a, e)
e = s.find("}  // namespace g13", a)
k = s.find("    __syncthreads();\n    int par = 0, j = 0;", a, e)
rep("    __syncthreads();\n    int par = 0, j = 0;", "    if (threadIdx.x == 0) TSR((KK == 8192 ? (n == 2304 ? 0 : 4) : 1), 1, xv[0].x ^ xv[0].w);\n    __syncthreads();\n    int par = 0, j = 0;", k, e)
e = s.find("}  // namespace g13", a)
k = s.find("        j += PF;\n    }\n}\n", a)
rep("        j += PF;\n    }\n}\n", "        j += PF;\n    }\n    __syncthreads();\n    if (threadIdx.x == 0) TSR((KK == 8192 ? (n == 2304 ? 0 : 4) : 1), 2, 1u);\n}\n", k, e)

# --- g14 (v1 qkv; same module as g13, so no second HDR): region 5 entry / end (S == 1 path) ---
a = s.find("_G14_SRC = r" + "'''")
A14 = "    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;\n"
k = s.find(A14, a)
rep(A14, A14 + "    if (t == 0) TSR(5, 0, 1u);\n", k)
k = s.find("            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(total(i) * scale);\n        }\n        return;\n", a)
rep("        }\n        return;\n", "        }\n        __syncthreads(); if (t == 0) TSR(5, 2, 1u);\n        return;\n", k)
# --- g30 (v1 o; own module): region 6 entry, CTA index x + gridDim.x * y ---
a = s.find("_G30_SRC = r" + "'''")
k = s.find("__global__ void __launch_bounds__(G30_GX", a)
rep("__global__ void __launch_bounds__(G30_GX", HDR.replace("+ blockIdx.x)", "+ blockIdx.x + gridDim.x * blockIdx.y)") + "__global__ void __launch_bounds__(G30_GX", k)
k = s.find('  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n', k)
rep('  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n', '  if (tid == 0) TSR(6, 0, 1u);\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n', k)

out = f"/tmp/x35_stamped_{os.getpid()}.py"
open(out, "w").write(open(src).read() if os.environ.get("NOSTAMP") else s)
sub = bench.load_sub(out)
w, st = bench.weights_state(sub)
(name, c), = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if f"{l} {c['layer']}" == flt]
d = bench.make_input(c, w, 99)
sub.kernel_fn(d, st); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    sub.kernel_fn(d, st)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
rows, tot = [], []
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for r in range(reps):
        flush.zero_(); buf.zero_(); torch.cuda.synchronize()
        ev0.record(); g.replay(); ev1.record(); torch.cuda.synchronize()
        tot.append(ev0.elapsed_time(ev1) * 1000)
        rows.append(buf.view(NR, NC, NK).cpu().clone())
print(name, f"layer replay median {statistics.median(tot):.1f} us (stamped copy)")
reg_names = {0: "qkv g13", 1: "o g13", 2: "csl", 3: "vtk", 4: "g13 K8k", 5: "qkv g14", 6: "o g30"}
slots = {0: {0: "entry", 1: "x arrived", 2: "end"}, 1: {0: "entry", 1: "x arrived", 2: "end"},
         2: {0: "entry", 9: "pre-PDL-wait", 10: "PDL wait ok", 1: "pt arrived", 8: "q arrived", 2: "KV arrived", 3: "QK+PV done",
             4: "syncthreads", 5: "cluster wait", 6: "merge rx ok", 7: "out stored"}}
slots[3] = slots[2]
slots[4] = slots[0]
slots[5] = {0: "entry", 2: "end"}
slots[6] = {0: "entry"}
lines = []
for reg in range(7):
    for k, nm in slots[reg].items():
        meds = []
        for t in rows:
            t0 = t[t > 0].min().item()  # earliest stamp of the replay (any kernel)
            v = t[reg, :, k]
            v = v[v > 0]
            if len(v) == 0: break
            v = (v - t0).double() / 1000.0
            meds.append((v.min().item(), v.median().item(), v.max().item(), len(v)))
        if len(meds) == len(rows):
            lines.append((statistics.median(m[0] for m in meds), f"{reg_names[reg]:8s} {nm:13s} n {meds[0][3]:3d}  min {statistics.median(m[0] for m in meds):7.2f}  med {statistics.median(m[1] for m in meds):7.2f}  max {statistics.median(m[2] for m in meds):7.2f} us"))
for _, l in sorted(lines):
    print(l)
os.remove(out)
