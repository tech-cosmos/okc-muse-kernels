# static tk2f timeline stamps -> muse_moe_w4a16/sub_tl.py (gitignored copy)
s=open('/home/daytona/r-moe/muse_moe_w4a16/submission.py').read()
def rep(a,b,n=1,start=0):
    global s
    i=s.find(a,start); assert i>=0,a
    s=s[:i]+b+s[i+len(a):]
rep('template <typename... KA, typename... AA>\nstatic int launch_k(', '''__device__ unsigned long long g_td[1024 * 8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TD(k) g_td[blockIdx.x * 8 + (k)] = gtime()
int64_t moe_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 8 * 8); return 0; }
template <typename... KA, typename... AA>
static int launch_k(''')
i=s.find('__device__ __forceinline__ void tk2f_consume(')
# stamp 2/5: first full wait done in phase 1 / phase 2 (consumer tid 0)
rep('''      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);''','''      mbar_wait(&full[st], ph);
      if (tid == 0 && R == r0 && q == 0) TD(GATED ? 2 : 5);
      const uint4* sw4 = wsm + st * (SB / 16);''',start=i)
rep('''        } while (c < need);''','''        } while (c < need);
        if (tid == 0 && R == r0) TD(4);''',start=i)
j=s.find('moe_tk2f(\n')
rep('''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int gen = 0;''','''  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TD(0);
  int gen = 0;''',start=j)
rep('''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];''','''  const int n13 = s_rs13[nact], n2 = s_rs2[nact];
  if (tid == 0) TD(1);''',start=j)
rep('''  tk2f_consume<256, NC, NSTG, false, KPS, SA>(slut, gs2, inter, Y, cnt, n13, b0, b1,''','''  if (tid == 0) TD(3);
  tk2f_consume<256, NC, NSTG, false, KPS, SA>(slut, gs2, inter, Y, cnt, n13, b0, b1,''',start=j)
rep('''  if constexpr (FN) {
    s_rt[tid] = rt;''','''  if (tid == 0) TD(6);
  if constexpr (FN) {
    s_rt[tid] = rt;''',start=j)
rep('"int64_t moe_set_pdl(int64_t, int64_t);"','"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
rep('functions=["moe_gemm13",','functions=["moe_tdbg", "moe_gemm13",')
open('/tmp/moe12/sub_tl.py','w').write(s)
# s12: H200 timeline diag: DIAG = 1 and _diag replaced by a timeline print
s = open('/tmp/moe12/sub_tl.py').read()
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
s = s.replace('        _diag(st)\n', '        _diag_tl(st)\n', 1)
s += r'''

def _diag_tl(st):
    import zlib, base64
    raw = zlib.decompress(base64.b64decode(_DIAG_R))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.empty((), dtype=torch.int64, device="cuda")
    o = torch.zeros(1024 * 8, dtype=torch.int64)
    names = ["start", "routed", "ph1 data", "ph1 end", "cnt ok", "ph2 data", "end"]
    p = 0
    for load, seed, T in _DIAG_HDR:
        loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda()
        p += T * 8
        if T > TK2_MAX:
            continue
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        data = {"hidden_states": torch.randn(T, H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
                "topk_weights": tw / tw.sum(-1, keepdim=True),
                "config": {"bs": T, "d_hidden": H, "local_start": st["start"], "local_count": st["count"]}}
        kernel_fn(data, st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            kernel_fn(data, st)
        acc, tot = [], []
        for _ in range(9):
            torch.sum(flush, 0, out=sink)
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); gr.replay(); b.record(); torch.cuda.synchronize()
            tot.append(a.elapsed_time(b) * 1e3)
            st["lib"].moe_tdbg(o)
            t = o.view(1024, 8)[:132, :7].double(); t = (t - t[:, 0].min()) / 1000
            acc.append(t)
        Tm = torch.stack(acc).median(0).values
        print(f"diagtl {load} s{seed} total {sorted(tot)[4]:.1f}: " + "  ".join(f"{n} {Tm[:, k].min():.1f}/{Tm[:, k].median():.1f}/{Tm[:, k].max():.1f}" for k, n in enumerate(names)), flush=True)
    del flush
'''
open('/tmp/moe12/submission.py', 'w').write(s)
