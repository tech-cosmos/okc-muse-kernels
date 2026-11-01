"""n65: per-CTA stamps of vjx's flat q_len-8 kernel (dev build): start (after partition), consumer end (before the
grid-wide finalize), kernel end; plus SM id, segments, keys. Shows the end-of-work spread the finalize waits on.
python dev/n65/stamps.py <batch> <l2|hbm> <q_len> [runs]"""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench, reference, vjx_src
B, mode, QL = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
S = vjx_src.CUDA_SRC
GT = "({ long long t_; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t_)); t_; })"
CI = "(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16"
a = "  int vc = c;\n"
assert S.count(a) == 1
S = S.replace(a, a + f"  if (p.dbg && threadIdx.x == 0) {{ unsigned sm_; asm(\"mov.u32 %0, %%smid;\" : \"=r\"(sm_)); "
              f"p.dbg[{CI} + 0] = {GT}; p.dbg[{CI} + 3] = sm_; p.dbg[{CI} + 5] = ce - cs; }}\n")
b = "      run_segments(std::integral_constant<int, 2>());\n"
assert S.count(b) == 1
S = S.replace(b, b + f"      if (p.dbg) {{ asm volatile(\"bar.sync 1, %0;\" :: \"n\"(Cfg<L, FLAT>::NT)); if (threadIdx.x == 0) {{ p.dbg[{CI} + 1] = {GT}; p.dbg[{CI} + 4] = nseg; p.dbg[{CI} + 2] = {GT}; }} }}\n")
vjx_src.CUDA_SRC = S
mod = vjx_src._build(tag="n65st")
w = reference.generate_weights(reference.WEIGHTS_SEED)
c = dict(batch=B, ctx=80000, layer="full", page=1, q_len=QL, dist="soft", spread="agentic")
data = bench.make_input(c, w, 7)
cfg = data["config"]
torch.manual_seed(0)
qkv = torch.randn(B, QL, reference.QKV_SIZE, device="cuda", dtype=torch.bfloat16)
pt = data["page_table"]
if mode == "l2":
    pt = torch.where(pt >= 0, pt % 8192, pt)
vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens")}
vd.update(qkv=qkv, config=cfg, page_table=pt)
dbg = torch.zeros(4096 * 16, dtype=torch.int64, device="cuda")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
import fcntl
res = []
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for r in range(runs + 1):
        flush.zero_()
        dbg.zero_()
        vjx_src._launch(vd, dbg=dbg, _mod=mod, pdl=False)
        torch.cuda.synchronize()
        if r:
            res.append(dbg.view(-1, 16)[:, :6].cpu().clone())
for d in res:
    n = int((d[:, 0] > 0).sum())
    d = d[:n].double()
    t0 = d[:, 0].min()
    st, ce, fe = (d[:, 0] - t0) / 1e3, (d[:, 1] - t0) / 1e3, (d[:, 2] - t0) / 1e3
    work = ce - st
    q = lambda x, f: float(x.quantile(f))  # noqa: E731
    print(f"CTAs {n}: start max {st.max():.1f}  cons-end min {ce.min():.1f} p10 {q(ce,.1):.1f} med {q(ce,.5):.1f} "
          f"p90 {q(ce,.9):.1f} max {ce.max():.1f}  kernel-end {fe.max():.1f}  mean wait {float((ce.max()-ce).mean()):.1f} us")
d = res[-1][:n].double()
ce = (d[:, 1] - d[:, 0].min()) / 1e3
order = ce.argsort(descending=True)[:12]
print("slowest CTAs: (cta, sm, nseg, keys, end us)")
print([(int(i), int(d[i, 3]), int(d[i, 4]), int(d[i, 5]), round(float(ce[i]), 1)) for i in order])
sm = d[:, 3]
nseg = d[:, 4]
for k in sorted(set(nseg.tolist())):
    m = nseg == k
    print(f"nseg {int(k)}: {int(m.sum())} CTAs, mean end {float(ce[m].mean()):.1f}, mean keys {float(d[m, 5].mean()):.0f}")
# per-SM stability: mean consumer-end per SM, correlation between runs
import numpy as np
per = []
for d in res:
    d = d[:n].double()
    ce = ((d[:, 1] - d[:, 0].min()) / 1e3).numpy()
    sm = d[:, 3].numpy().astype(int)
    v = np.zeros(sm.max() + 1); k = np.zeros(sm.max() + 1)
    np.add.at(v, sm, ce); np.add.at(k, sm, 1)
    per.append(v / np.maximum(k, 1))
per = np.array(per)
cc = np.corrcoef(per)
print("per-SM end corr between runs:", np.round(cc[np.triu_indices(len(per), 1)], 2))
m = per.mean(0)
print("per-SM mean end: spread min %.1f max %.1f; slowest SMs" % (m.min(), m.max()), np.argsort(-m)[:10], "fastest", np.argsort(m)[:10])
# GPC-ish structure: print mean end by SM id in rows of 12
print(np.round(m, 0).reshape(-1, 12) if len(m) % 12 == 0 else np.round(m, 0))
d = res[-1][:n].double()
ce = ((d[:, 1] - d[:, 0].min()) / 1e3).numpy()
print("end by blockIdx (rows of 24):")
print(np.round(ce, 0).reshape(-1, 24) if n % 24 == 0 else np.round(ce, 0))
sm = d[:, 3].numpy().astype(int)
pair_fast = [ce[sm == s].min() for s in range(sm.max() + 1)]
pair_slow = [ce[sm == s].max() for s in range(sm.max() + 1)]
print("per-SM min/max mean", np.mean(pair_fast), np.mean(pair_slow))
