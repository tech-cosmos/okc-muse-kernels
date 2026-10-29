# n7s22: H200 per-CTA timeline DIAG of the current decode path (moe_tk2f + moe_norm2/3), layer routings (T <= 8 by default).
# usage: python tools/n7s22/mktl.py [maxT] -> /tmp/n7s22/submission.py (okc check it; grep diagtl in the job log)
import os, sys
os.makedirs("/tmp/n7s22", exist_ok=True)
MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 8
s = open(os.environ.get("SRC", os.path.join(os.path.dirname(__file__), "../../submission.py"))).read()
def rep(a, b, start=0, uniq=True):
    global s
    i = s.find(a, start); assert i >= 0, a
    if uniq: assert s.find(a, i + 1) < 0 or start, ("not unique", a)
    s = s[:i] + b + s[i + len(a):]
    return i
rep('template <typename... KA, typename... AA>\nstatic int launch_k(', '''__device__ unsigned long long g_td[1024 * 8];
__device__ unsigned long long g_tn[4];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TD(k) g_td[blockIdx.x * 8 + (k)] = gtime()
int64_t moe_tdbg(torch::Tensor o) {
  cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 8 * 8);
  cudaMemcpyFromSymbol((char*)o.data_ptr() + 1024 * 64, g_tn, 32);
  unsigned long long z[4] = {~0ull, ~0ull, 0ull, 0ull};
  cudaMemcpyToSymbol(g_tn, z, 32);
  return 0;
}
template <typename... KA, typename... AA>
static int launch_k(''')
# norm2 / norm3: first CTA start, first post-wait, last end (atomics)
for k in ("moe_norm2(",):
    i = s.find("__global__ void __launch_bounds__(512) " + k); assert i >= 0, k
    j = s.find('asm volatile("griddepcontrol.wait;" ::: "memory");', i)
    s = s[:j] + 'if (threadIdx.x == 0) atomicMin(&g_tn[0], gtime());\n  ' + s[j:j + 51] + '\n  if (threadIdx.x == 0) atomicMin(&g_tn[1], gtime());' + s[j + 51:]
    # end: before the kernel's closing brace = next "\n}\n" after j
    e = s.find("\n}\n", j)
    s = s[:e] + '\n  __syncthreads();\n  if (threadIdx.x == 0) atomicMax(&g_tn[2], gtime());' + s[e:]
ic = s.find('__device__ __forceinline__ void tk2f_consume(')
rep('''      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);''', '''      mbar_wait(&full[st], ph);
      if (tid == 0 && R == r0 && q == 0) TD(GATED ? 2 : 5);
      const uint4* sw4 = wsm + st * (SB / 16);''', start=ic)
rep('''      const bool v = g < n;
      int row = 0;''', '''      if (!GATED && tid == 0 && R == r0) TD(4);
      const bool v = g < n;
      int row = 0;''', start=ic)
ik = s.find('moe_tk2f(\n    const uint4* __restrict__ Wq13')
rep('''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr bool FP = (PF & 16) != 0;''', '''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TD(0);
  constexpr bool FP = (PF & 16) != 0;''', start=ik)
rep('''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];''', '''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];
  if (tid == 0) TD(1);''', start=ik)
rep('''      tk2f_produce<512, NC, NSTG, KPS, SQ, S6, S6P>(Wq13, S13, a0, a1, s_rs13, s_e, smem, full, empty, st, ph, i, nullptr, 0, l2hv, nullptr, PFD);''',
    '''      TD(7);
      tk2f_produce<512, NC, NSTG, KPS, SQ, S6, S6P>(Wq13, S13, a0, a1, s_rs13, s_e, smem, full, empty, st, ph, i, nullptr, 0, l2hv, nullptr, PFD);''', start=ik)
rep('''  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, s_red, smem, full,
                                     empty, st, ph, rb);''', '''  if (tid == 0) TD(3);
  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, s_red, smem, full,
                                     empty, st, ph, rb);
  if (tid == 0) TD(6);''', start=ik)
rep('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
rep('functions=["moe_gemm13",', 'functions=["moe_tdbg", "moe_gemm13",')
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
s = s.replace("        _diag(st)\n", "        _diag_tl(st)\n" + ("        _diag(st)\n" if os.environ.get("BOTH") else ""), 1)
s += r'''

def _diag_tl(st):
    import zlib, base64
    raw = zlib.decompress(base64.b64decode(_DIAG_R))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.empty((), dtype=torch.int64, device="cuda")
    o = torch.zeros(1024 * 8 + 4, dtype=torch.int64)
    lib = _lib()
    names = ["start", "routed", "ph1data", "ph1end", "cntok", "ph2data", "end", "prod1st"]
    p = 0
    lib.moe_tdbg(o)
    for load, seed, T in _DIAG_HDR:
        loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda()
        p += T * 8
        if T > MAXT_TL:
            continue
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        nexp = int(torch.unique(loc[loc >= 0]).numel())
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        data = {"hidden_states": torch.randn(T, H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
                "topk_weights": tw / tw.sum(-1, keepdim=True),
                "config": {"bs": T, "d_hidden": H, "local_start": st["start"], "local_count": st["count"]}}
        kernel_fn(data, st)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            kernel_fn(data, st)
        gr.replay(); torch.cuda.synchronize(); lib.moe_tdbg(o)
        acc, tot, nrm = [], [], []
        for _ in range(15):
            torch.sum(flush, 0, out=sink)
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); gr.replay(); b.record(); torch.cuda.synchronize()
            tot.append(a.elapsed_time(b) * 1e3)
            lib.moe_tdbg(o)
            t = o[:8192].view(1024, 8)[:132].double()
            t0 = t[:, 0].min()
            t = (t - t0) / 1000
            acc.append(t)
            nrm.append(((o[8192:8195].double() - t0) / 1000))
        Tm = torch.stack(acc).median(0).values
        Nm = torch.stack(nrm).median(0).values
        print(f"diagtl {load} s{seed} T{T} E{nexp} total {sorted(tot)[7]:.1f}: " + "  ".join(f"{n} {Tm[:, k].min():.2f}/{Tm[:, k].median():.2f}/{Tm[:, k].max():.2f}" for k, n in enumerate(names))
              + f"  norm launch {Nm[0]:.2f} wait {Nm[1]:.2f} end {Nm[2]:.2f}", flush=True)
        # per-CTA end histogram (0.5 us bins) of 'end' and 'ph1end'
        for k in (3, 6):
            h = torch.histc(Tm[:, k], bins=12, min=float(Tm[:, k].min()), max=float(Tm[:, k].max()) + 1e-6)
            print(f"diagtl   hist {names[k]} {Tm[:, k].min():.2f}..{Tm[:, k].max():.2f}: {h.int().tolist()}", flush=True)
    del flush
'''
s = s.replace("\ndef _diag_tl(st):", f"\nMAXT_TL = {MAXT}\n\ndef _diag_tl(st):", 1)
open('/tmp/n7s22/submission.py', 'w').write(s)
print("ok", len(s))
