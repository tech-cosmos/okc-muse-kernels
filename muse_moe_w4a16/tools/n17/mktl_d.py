# n17: H200 per-CTA timeline of moe_tk2f at d16/d32 (layer routings): gs8 SQ (main) vs tk2h GS16 SQ (TK2H_M W 0/3),
# kernel-level globaltimer stamps, as a DIAG check copy:
#   python tools/n17/mktl_d.py [out=/tmp/moe17/tl/submission.py]; okc check <out> --json; grep diagtd <job log>
# Columns (us from the earliest CTA entry, min/median/max over CTAs of the median of 9 cold-L2 replays):
# c0 = consumer start (routing + split done), p1 = gemm13 rows done, w = gemm2 counter spin, e = CTA end;
# n13/n2 = rows per CTA (min/max); tot = graph replay incl. norm.
import os, sys
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moe17/tl/submission.py"
os.makedirs(os.path.dirname(out), exist_ok=True)
s = open(os.path.join(os.path.dirname(__file__), "..", "..", "submission.py")).read()
def rep(a, b, start=0, cnt=1):
    global s
    for _ in range(cnt):
        i = s.find(a, start); assert i >= 0, a
        s = s[:i] + b + s[i + len(a):]
        start = i + len(b)
    return start
rep('template <typename... KA, typename... AA>\nstatic int launch_k(', '''__device__ unsigned long long g_td[1024 * 8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TD(k) g_td[blockIdx.x * 8 + (k)] = gtime()
int64_t moe_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 8 * 8); return 0; }
template <typename... KA, typename... AA>
static int launch_k(''')
# tk2f_consume: stamp the end of the gemm2 counter spin at the CTA's first gemm2 row
i = s.find('__device__ __forceinline__ void tk2f_consume(')
rep('''        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);
      }''', '''        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);
        if (threadIdx.x == 0 && R == r0) TD(3);
      }''', i)
# tk2h_consume: phase-2 start + spin end
i = s.find('__device__ __forceinline__ void tk2h_consume(')
j = rep('''    const int r0 = gated ? a0 : b0, r1 = gated ? a1 : b1;''', '''    const int r0 = gated ? a0 : b0, r1 = gated ? a1 : b1;
    if (threadIdx.x == 0 && !gated) TD(2);''', i)
rep('''          } while (c < nd);''', '''          } while (c < nd);
          if (threadIdx.x == 0 && R == r0) TD(3);''', j)
# moe_tk2f: entry, consumer start, phase 1 end (tk2f path), end, row counts
k = s.find('__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2f(')
rep('''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr bool FP = (PF & 16) != 0;''', '''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TD(1);
  constexpr bool FP = (PF & 16) != 0;''', k)
rep('''  int rb = 0;
  if constexpr (MRG) {
    tk2h_consume''', '''  int rb = 0;
  if (tid == 0) { TD(0); g_td[blockIdx.x * 8 + 6] = (unsigned long long)(a1 - a0) | ((unsigned long long)(b1 - b0) << 32); }
  if constexpr (MRG) {
    tk2h_consume''', k)
rep('''s_srt, s_red, smem, smem + NSTG * NC * KPS * 72, full, empty, st, ph, rb);
    return;''', '''s_srt, s_red, smem, smem + NSTG * NC * KPS * 72, full, empty, st, ph, rb);
    if (tid == 0) TD(5);
    return;''', k)
rep('''                                    empty, st, ph, rb, &xrdy);
  tk2f_consume<256''', '''                                    empty, st, ph, rb, &xrdy);
  if (tid == 0) TD(2);
  tk2f_consume<256''', k)
rep('''                                     empty, st, ph, rb);
  if constexpr (FN) {''', '''                                     empty, st, ph, rb);
  if (tid == 0) TD(5);
  if constexpr (FN) {''', k)
rep('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
rep('functions=["moe_gemm13",', 'functions=["moe_tdbg", "moe_gemm13",')
rep('\nDIAG = 0\n', '\nDIAG = 1\n')
rep('        _diag(st)\n', '        _diag_td(st)\n')
s += r'''

def _diag_td(st):
    import zlib, base64
    raw = zlib.decompress(base64.b64decode(_DIAG_R))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.empty((), dtype=torch.int64, device="cuda")
    o = torch.zeros(1024 * 8, dtype=torch.int64)
    g = globals()
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    p = 0
    for load, seed, T in _DIAG_HDR:
        loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda()
        p += T * 8
        if T not in (16, 32):
            continue
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        data = {"hidden_states": torch.randn(T, H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
                "topk_weights": tw / tw.sum(-1, keepdim=True),
                "config": {"bs": T, "d_hidden": H, "local_start": st["start"], "local_count": st["count"]}}
        for name, v in (("sq", {}), ("h0", {"TK2H_M": {16: 0, 32: 0}}), ("h3", {"TK2H_M": {16: 3, 32: 3}})):
            old = {k: g[k] for k in v}
            g.update(v)
            try:
                kernel_fn(data, st)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    kernel_fn(data, st)
            finally:
                g.update(old)
            acc, tot = [], []
            for _ in range(9):
                torch.sum(flush, 0, out=sink)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); gr.replay(); b.record(); torch.cuda.synchronize()
                tot.append(a.elapsed_time(b) * 1e3)
                st["lib"].moe_tdbg(o)
                t = o.view(1024, 8)[:nsm].clone()
                rc = t[:, 6].clone()
                t = t.double(); t = (t - t[:, 1].min()) / 1000
                acc.append(t)
            Tm = torch.stack(acc).median(0).values
            q = lambda x: f"{x.min():.1f}/{x.median():.1f}/{x.max():.1f}"
            n13, n2 = rc & 0xffffffff, rc >> 32
            print(f"diagtd {load[-2:]} s{seed} {name} tot {sorted(tot)[4]:.1f} c0 {q(Tm[:, 0])} p1 {q(Tm[:, 2])} "
                  f"w {q(Tm[:, 3] - Tm[:, 2])} e {q(Tm[:, 5])} r {n13.min()}-{n13.max()}/{n2.min()}-{n2.max()}", flush=True)
            del gr
    del flush
'''
open(out, "w").write(s)
print("wrote", out)
