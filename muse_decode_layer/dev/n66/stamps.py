"""n66: per-CTA stamps of the dynamic-pool build: start, consumer end, nseg, pool tiles drained, sentinel time.
python dev/n66/stamps.py <batch> <dynf> [dynch]"""
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "dev/n66")
import bench, reference, vjx_src, dyn, donor
B, F = int(sys.argv[1]), int(sys.argv[2]); CH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
S = vjx_src.CUDA_SRC
GT = "({ long long t_; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t_)); t_; })"
CI = "(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16"
a = "  int vc = c;\n"
S = S.replace(a, a + f"  if (p.dbg && threadIdx.x == 0) {{ unsigned sm_; asm(\"mov.u32 %0, %%smid;\" : \"=r\"(sm_)); "
              f"p.dbg[{CI} + 0] = {GT}; p.dbg[{CI} + 3] = sm_; }}\n")
b = "      run_segments(std::integral_constant<int, 2>());\n"
S = S.replace(b, b + f"      if (p.dbg) {{ asm volatile(\"bar.sync 1, %0;\" :: \"n\"(Cfg<L, FLAT>::NT)); if (threadIdx.x == 0) {{ p.dbg[{CI} + 1] = {GT}; p.dbg[{CI} + 4] = nseg; p.dbg[{CI} + 11] = S.seg[nseg - 1][0]; }} }}\n")
S = donor.patch(S, -F, CH) if F < 0 else dyn.patch(S, F, CH)
c = "        {   // sentinel stage: no copies, flag 0\n"
assert S.count(c) == 1
S = S.replace(c, f"        if (p.dbg && q == 0) {{ p.dbg[{CI} + 5] = x - ntiles; p.dbg[{CI} + 7] = {GT}; p.dbg[{CI} + 8] = ntiles; }}\n" + c)
e = "          if (__any_sync(0xffffffffu, ov != 0u)) {\n"
assert S.count(e) == 1, S.count(e)
S = S.replace(e, e + f"            if (p.dbg && (threadIdx.x & 31) == 0) atomicAdd((unsigned long long*)&p.dbg[{CI} + 9 + (i >= ntiles ? 1 : 0)], 1ull);\n")
vjx_src.CUDA_SRC = S
mod = vjx_src._build(tag=f"n66st{F}_{CH}".replace("-", "m"))
w = reference.generate_weights(reference.WEIGHTS_SEED)
cc = dict(batch=B, ctx=80000, layer="full", page=1, q_len=8, dist="soft", spread="agentic")
data = bench.make_input(cc, w, 7)
torch.manual_seed(0)
qkv = torch.randn(B, 8, reference.QKV_SIZE, device="cuda", dtype=torch.bfloat16)
vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens", "page_table")}
vd.update(qkv=qkv, config=data["config"])
dbg = torch.zeros(4096 * 16, dtype=torch.int64, device="cuda")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
import fcntl
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for r in range(3):
        flush.zero_(); dbg.zero_()
        vjx_src._launch(vd, dbg=dbg, _mod=mod, pdl=False)
        torch.cuda.synchronize()
d = dbg.view(-1, 16)[:, :12].cpu().double()
n = int((d[:, 0] > 0).sum()); d = d[:n]
t0 = d[:, 0].min()
st, ce = (d[:, 0] - t0) / 1e3, (d[:, 1] - t0) / 1e3
drn = d[:, 7] > 0
sen = torch.where(drn, (d[:, 7] - t0) / 1e3, torch.zeros_like(st))
print(f"CTAs {n}: start max {st.max():.1f}; cons-end min {ce.min():.1f} med {ce.median():.1f} max {ce.max():.1f}; mean wait {float((ce.max()-ce).mean()):.1f}")
if drn.any(): print(f"drainers {int(drn.sum())}; pool tiles taken: total {int(d[:,5].sum())}, per drainer min {int(d[drn,5].min())} med {float(d[drn,5].median()):.0f} max {int(d[drn,5].max())}")
if drn.any(): print(f"static tiles (drainers): min {int(d[drn,8].min())} max {int(d[drn,8].max())}; sentinel time min {float(sen[drn].min()):.1f} max {float(sen[drn].max()):.1f}")
order = ce.argsort(descending=True)[:10]
print("slowest (cta, nseg, drained, static tiles, sentinel us, end us):", [(int(i), int(d[i,4]), int(d[i,5]), int(d[i,8]), round(float(sen[i]),1), round(float(ce[i]),1)) for i in order])
print("offset-move hits: static tiles", int(d[:,9].sum()), "pool tiles", int(d[:,10].sum()))
nd = ~drn
print(f"non-drainers {int(nd.sum())}: end med {float(ce[nd].median()) if nd.any() else 0:.1f} max {float(ce[nd].max()) if nd.any() else 0:.1f}")

import collections
grp = collections.defaultdict(list)
for i in range(n):
    grp[int(d[i, 11])].append(float(ce[i]))
gm = sorted((max(v), min(v), len(v), k) for k, v in grp.items())
print("per last-request group (max end, min end, nCTAs, b):", [(round(a, 1), round(b_, 1), c_, k) for a, b_, c_, k in gm])
