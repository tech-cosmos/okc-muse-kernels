"""n13s1 d32 qkv oracle (H100, standalone, cold L2): the layer's g30n qkv_gate m32 pick (MTOK 32, S 3, NW 16, NS 16)
rebuilt with per-CTA globaltimer stamps. python tools/n13/qkv_stamp.py <submission.py> [cfg MTOK,S,NW,NS] [reps]
Slots: 0 start, 1/2 first stage (W+x) ready WG0/WG1, 3/4 loop end WG0/WG1, 5 CTA reduce done, 6 cluster reduce done
(rank 0), 7 end (rank 0 after stores; ranks > 0 after the DSMEM push)."""
import sys, os, statistics, fcntl, hashlib, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
cfg = tuple(int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "32,3,16,16").split(","))
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
VAR = os.environ.get("VAR", "")  # nomath: skip decode+wgmma (stream only)
w, st = bench.weights_state(sub)
src = sub._G30_SRC
ST = "asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(_gt)); g30_st[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + (K)] = _gt;"
hdr = ("__device__ unsigned long long g30_st[4096 * 16]; __device__ unsigned g30_sm[4096];\n"
       "#define G30ST(K) { unsigned long long _gt; " + ST + " }\n")
def rep1(s, a, b):
    assert a in s, a[:60]
    return s.replace(a, b, 1)
src = rep1(src, "namespace g30 {", hdr + "namespace g30 {")
src = rep1(src, '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  if (tid == 0) {',
           '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  if (tid == 0) { unsigned _s; asm("mov.u32 %0, %%smid;" : "=r"(_s)); '
           'g30_sm[blockIdx.y * gridDim.x + blockIdx.x] = _s; G30ST(0) }\n  if (tid == 0) {')
src = rep1(src, "        mbar_wait(&wempty[st], ph ^ 1);\n", "        if (i == 0) G30ST(8)\n        mbar_wait(&wempty[st], ph ^ 1);\n")
src = rep1(src, "        mbar_wait(&empty[st], ph ^ 1);\n", "        if (i == 0) G30ST(10)\n        mbar_wait(&empty[st], ph ^ 1);\n")
src = rep1(src, "  int i = i0 + wg;\n  while (i < i1) {", "  if (t == 0 && wg == 0) G30ST(9)\n  int i = i0 + wg;\n  while (i < i1) {")
src = rep1(src, "    mbar_wait(&full[st], ph);                                                                               \\\n",
           "    mbar_wait(&full[st], ph);                                                                               \\\n"
           "    if (ir < 2 && t == 0) G30ST(1 + wg)                                                                     \\\n")
if "early" in VAR:  # cluster barrier: arrive at start (after a CTA sync), wait only before the DSMEM push / rbar wait
    src = rep1(src, '#if G30_S > 1\n  asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n#else',
               '#if G30_S > 1\n  __syncthreads();\n  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");\n#else')
    src = rep1(src, "  if (blockIdx.y > 0) {  // push the partial", '  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");\n  if (blockIdx.y > 0) {  // push the partial')
if "nored" in VAR:  # timing only: no DSMEM push, rank 0 does not wait
    src = rep1(src, "  if (blockIdx.y > 0) {  // push the partial", "  if (blockIdx.y > 0) { if (t == 0) G30ST(7) return; }\n  if (0) {  // push the partial")
    src = rep1(src, '        " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\\n"', '        " mov.pred p, 1;\\n"')
if "rot" in VAR:  # timing only: tile g = (blockIdx.x + 17) % gridDim.x (W/x/out move to other SMs)
    src = rep1(src, "  const int g = blockIdx.x;\n  const int i0 = 2 * ((int)blockIdx.y", "  const int g = (blockIdx.x + 17) % gridDim.x;\n  const int i0 = 2 * ((int)blockIdx.y")
if "nomath" in VAR:
    src = rep1(src, "    G30_DEC(a, 0, s01.x, s01.y, nw.x) G30_DEC(a, 1, s01.z, s01.w, nw.y) G30_DEC(a, 2, s23.x, s23.y, nw.z)   \\\n"
               "    G30_DEC(a, 3, s23.z, s23.w, nw.w) G30_FIX(a)                                                            \\\n",
               "    a[0] ^= s01.x ^ s23.y ^ nw.z;                                                                             \\\n")
    src = rep1(src, "    _Pragma(\"unroll\") for (int j = 0; j < 4; j++) wg_mma_rs(d, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], dsw128(b + j * 32)); \\\n",
               "    d[0] += __uint_as_float(a[0] & 0x3f) + (float)b;                                                        \\\n")
src = rep1(src, '  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n#pragma unroll\n  for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));',
           '  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n  if (t == 0) G30ST(3 + wg)\n#pragma unroll\n  for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));')
src = rep1(src, "  if (wg == 1) return;\n#pragma unroll\n  for (int j = 0; j < G30_ND; j++) d[j] += red[j * 128 + t];",
           "  if (wg == 1) return;\n  if (t == 0) G30ST(5)\n#pragma unroll\n  for (int j = 0; j < G30_ND; j++) d[j] += red[j * 128 + t];")
src = rep1(src, "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\" :: \"r\"(rm) : \"memory\");\n    return;",
           "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\" :: \"r\"(rm) : \"memory\");\n    if (t == 0) G30ST(7)\n    return;")
src = rep1(src, "  // acc layout (m64n128)", "  if (t == 0) G30ST(6)\n  // acc layout (m64n128)")
src = rep1(src, "*(uint4*)(out + (size_t)tk * N + g * 64 + c * 8) = *(const uint4*)(stg + tk * 72 + c * 8);\n  }\n}",
           "*(uint4*)(out + (size_t)tk * N + g * 64 + c * 8) = *(const uint4*)(stg + tk * 72 + c * 8);\n  }\n  if (t == 0) G30ST(7)\n}")
src += """
void g30_st_copy(int64_t dst, int64_t dsm) {
  cudaMemcpyFromSymbol((void*)dst, g30_st, sizeof(g30_st), 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyFromSymbol((void*)dsm, g30_sm, sizeof(g30_sm), 0, cudaMemcpyDeviceToDevice);
}
void g30_st_clear() { static unsigned long long z[4096 * 16]; cudaMemcpyToSymbol(g30_st, z, sizeof(z)); }
"""
old = sub._G30_SRC
sub._G30_SRC = src
# expose the stamp copy fn via the TSHIM loader: add it to the functions list
from torch.utils import cpp_extension
orig = cpp_extension.load_inline
def li(**kw):
    kw["functions"] = list(kw["functions"]) + ["g30_st_copy", "g30_st_clear"]
    kw["cpp_sources"] += "\nvoid g30_st_copy(int64_t, int64_t);\nvoid g30_st_clear();\n"
    return orig(**kw)
cpp_extension.load_inline = li
dfs = (["-DG30_ESC=0"] if "noesc" in VAR else []) + [f"-DG30_MTOK={cfg[0]}", f"-DG30_S={cfg[1]}", f"-DG30_NW={cfg[2]}", f"-DG30_NS={cfg[3]}"]
lib = sub._g30_load("n13_g30st_" + VAR.replace(",", "_") + hashlib.sha256((src + " ".join(dfs)).encode()).hexdigest()[:10], dfs, False)
cpp_extension.load_inline = orig
_, e, N = st["g30n"][("qkv_gate", 32)]
m = cfg[0]
K = w["qkv_gate"].shape[1]
x = torch.randn((m, K), device="cuda").bfloat16()
out = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")
lib.g30_gemm(x, e[0], e[1], out, e[2], 0)
torch.cuda.synchronize()
if VAR != "nomath":
    ref = torch.nn.functional.linear(x, w["qkv_gate"]).float()
    print("rel err", ((out.float() - ref).norm() / ref.norm()).item())
fn = lambda: lib.g30_gemm(x, e[0], e[1], out, e[2], 0)
t = bench.time_graph(fn, reps) - bench.empty_cost()
G = (N // 64) * cfg[1]
stb = torch.zeros(4096 * 16, dtype=torch.int64, device="cuda"); smb = torch.zeros(4096, dtype=torch.int32, device="cuda")
# stamps of one more cold replay (graph replay, after flush), several samples
g = bench.capture(fn)
rows = []
for r in range(7):
    torch.sum(bench._FLUSH.view(torch.int64), 0, out=bench._SINK)
    lib.mod.g30_st_clear(); torch.cuda.synchronize()
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        g.replay(); torch.cuda.synchronize()
    lib.mod.g30_st_copy(stb.data_ptr(), smb.data_ptr()); torch.cuda.synchronize()
    s = stb[: G * 16].view(G, 16).cpu().double()
    rows.append(s)
pb = e[0].numel() * e[0].element_size() / G
print(f"cfg {cfg} VAR={VAR!r}: CTAs {G}, W bytes/CTA {pb/1e3:.0f} KB, total W {e[0].numel()*e[0].element_size()/1e6:.1f} MB; graph time {t:.2f} us")
def q(v):
    v = sorted(v); return f"{v[0]:6.2f} {v[len(v)//2]:6.2f} {v[-1]:6.2f}"
agg = {}
for s in rows:
    t0 = s[:, 0].min()
    r = (s - t0) / 1e3
    y = torch.arange(G) // (N // 64)
    agg.setdefault("start", []).append(q(r[:, 0].tolist()))
    agg.setdefault("W producer 1st issue", []).append(q(r[:, 8].tolist()))
    agg.setdefault("x producer 1st issue", []).append(q(r[:, 10].tolist()))
    agg.setdefault("consumer prologue done", []).append(q(r[:, 9].tolist()))
    agg.setdefault("first W+x WG0", []).append(q(r[:, 1].tolist()))
    agg.setdefault("loop end WG0", []).append(q(r[:, 3].tolist()))
    agg.setdefault("loop end WG1", []).append(q(r[:, 4].tolist()))
    agg.setdefault("CTA red done", []).append(q(r[:, 5].tolist()))
    r0 = r[y == 0]
    agg.setdefault("rank0 cluster red done", []).append(q(r0[:, 6].tolist()))
    agg.setdefault("end (all)", []).append(q(r[:, 7].tolist()))
    agg.setdefault("rate GB/s/CTA (loop)", []).append(q((pb / ((r[:, 3:5].max(1).values - r[:, 1]) * 1e3)).tolist()))
print("stamp (us from first CTA start)  min med max  [per replay]")
for k, v in agg.items():
    print(f"  {k:24s} " + " | ".join(v[:4]))
sm = smb[:G].cpu().tolist()
print("distinct SMs", len(set(sm)))
# n13s1 tail analysis: per-CTA loop time vs TPC sibling occupancy, cluster balance, repeatability
smid = torch.tensor(sm)
occ = set(sm)
sib = torch.tensor([(s ^ 1) in occ for s in sm])
L = torch.stack([((r[:, 3:5].max(1).values - r[:, 8]) / 1e3) for r in [(s - s[:, 0].min()) for s in rows]])  # [reps, G] us
mL = L.mean(0)
print(f"loop us (W issue -> loop end), mean over replays: CTAs with TPC sibling busy {mL[sib].mean():.2f} (n={int(sib.sum())}), alone {mL[~sib].mean():.2f} (n={int((~sib).sum())})")
print("replay-to-replay corr of per-CTA loop time:", f"{torch.corrcoef(L)[0, 1:].mean():.2f}")
y = torch.arange(G) // (N // 64); xg = torch.arange(G) % (N // 64)
cl = mL.view(cfg[1], -1)  # [rank, tile]
print(f"cluster: mean of per-cluster max {cl.max(0).values.mean():.2f}, max {cl.max().item():.2f}, mean {cl.mean():.2f}, max of cluster means {cl.mean(0).max():.2f}")
print("per rank mean", [f"{v:.2f}" for v in cl.mean(1).tolist()])
order = mL.argsort()
gt = (xg + (17 if "rot" in VAR else 0)) % (N // 64)
print("slowest 12 (tile, rank, smid, us):", [(int(gt[i]), int(y[i]), sm[i], round(float(mL[i]), 2)) for i in order[-12:].tolist()])
print("fastest 6:", [(int(gt[i]), int(y[i]), sm[i], round(float(mL[i]), 2)) for i in order[:6].tolist()])
