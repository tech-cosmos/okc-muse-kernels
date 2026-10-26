# n4s5: H200 per-CTA timeline of tk2h (TK2H 8 / 16 x TK2H_W) on the layer's d16/d32 routings, as a DIAG check copy:
#   python tools/n4s5/mktl_h.py [out=/tmp/moe5tl/submission.py]; okc check <out> --json; grep diagth <job log>
# Columns (us from the earliest CTA start, min/median/max over CTAs): c0 consumer start (after routing + split),
# ph1 end (gemm13 rows done), wait (gemm2 counter spin), end; total = graph replay incl. norm2.
import os, sys
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moe5tl/submission.py"
os.makedirs(os.path.dirname(out), exist_ok=True)
s = open(os.path.join(os.path.dirname(__file__), "..", "..", "submission.py")).read()
def rep(a, b, start=0):
    global s
    i = s.find(a, start); assert i >= 0, a
    s = s[:i] + b + s[i + len(a):]
rep('template <typename... KA, typename... AA>\nstatic int launch_k(', '''__device__ unsigned long long g_td[1024 * 8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TD(k) g_td[blockIdx.x * 8 + (k)] = gtime()
int64_t moe_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 8 * 8); return 0; }
template <typename... KA, typename... AA>
static int launch_k(''')
i = s.find('__device__ __forceinline__ void tk2h_consume(')
rep('''  for (int p = 0; p < 2; ++p) {
    const bool gated = p == 0;''', '''  for (int p = 0; p < 2; ++p) {
    if (tid == 0) TD(p ? 2 : 0);
    const bool gated = p == 0;''', i)
rep('''          } while (c < nd);''', '''          } while (c < nd);
          if (tid == 0 && R == r0) TD(3);''', i)
j = s.find('tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2>(')
rep('''    return;
  }
  if constexpr (GS == 16) {''', '''    if (tid == 0) TD(5);
    return;
  }
  if constexpr (GS == 16) {''', j)
rep('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
rep('functions=["moe_gemm13",', 'functions=["moe_tdbg", "moe_gemm13",')
rep('\nDIAG = 0\n', '\nDIAG = 1\n')
rep('        _diag(st)\n', '        _diag_th(st)\n')
s += r'''

def _diag_th(st):
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
        if T > TK2_MAX or T < 9:
            continue
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        data = {"hidden_states": torch.randn(T, H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
                "topk_weights": tw / tw.sum(-1, keepdim=True),
                "config": {"bs": T, "d_hidden": H, "local_start": st["start"], "local_count": st["count"]}}
        for name, v in (("h8", {"TK2H": 8}), ("h16w0", {"TK2H": 16}), ("h16w2", {"TK2H": 16, "TK2H_W": 2}), ("h16w3", {"TK2H": 16, "TK2H_W": 3})):
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
                t = o.view(1024, 8)[:nsm].double(); t = (t - t[:, 0].min()) / 1000
                acc.append(t)
            Tm = torch.stack(acc).median(0).values
            q = lambda x: f"{x.min():.1f}/{x.median():.1f}/{x.max():.1f}"
            print(f"diagth {load} s{seed} {name} total {sorted(tot)[4]:.1f}: c0 {q(Tm[:, 0])} ph1end {q(Tm[:, 2])} "
                  f"wait {q(Tm[:, 3] - Tm[:, 2])} end {q(Tm[:, 5])}", flush=True)
            del gr
    del flush
'''
open(out, "w").write(s)
print("wrote", out)
