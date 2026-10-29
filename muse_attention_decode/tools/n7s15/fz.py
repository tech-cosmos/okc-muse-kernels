"""n7s15 gate: fused inversion + dense K/V stream (fz.cu). Usage: python tools/n7s15/fz.py"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import bench, reference
from torch.utils.cpp_extension import load_inline
m = load_inline(name="n7s15_fz", cpp_sources="void fz_run(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources=open(os.path.join(os.path.dirname(__file__), "fz.cu")).read(), functions=["fz_run"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-Xptxas=-v"], verbose=False)
T = lambda f: min(bench.time_graph(f) - bench.empty_cost() for _ in range(3))
NW = 132
E = torch.empty(0, dtype=torch.int64, device="cuda")


def magic(ws):
    p = torch.arange(1 << 22, device="cuda", dtype=torch.int64)
    for sh in range(0, 12):
        mg = -(-(1 << (32 + sh)) // ws)
        if mg >= 1 << 32: break
        if torch.equal((p * mg) >> (32 + sh), p // ws): return mg, sh
    raise RuntimeError


for batch, S, per in ((32, 4, 28), (16, 8, 14)):
    d = reference.generate_input(batch=batch, ctx=80000, heads="tp8", layer="full", seed=99, page=1, q_len=1, dist="soft", spread="agentic")
    pt, seq = d["page_table"], d["seq_lens"].to(torch.int32)
    kc = d["k_cache"].view(torch.uint8).reshape(-1, 128); vc = d["v_cache"].view(torch.uint8).reshape(-1, 128)
    rows = kc.shape[0]; ws = (rows + NW - 1) // NW; mg, sh = magic(ws)
    ent = per * 768; assert S * ent >= pt.shape[1]
    out = torch.zeros(batch * S * ent, dtype=torch.int16, device="cuda"); off = torch.zeros(batch * S * (NW + 1), dtype=torch.int16, device="cuda")
    cnt = torch.zeros(4, dtype=torch.int32, device="cuda"); sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    # correctness of the inversion: rebuild the map from out/off on the host
    m.fz_run(kc, vc, ws, mg, sh, pt, seq, S, out, off, cnt, sink, E, per, 1, 256, 2); torch.cuda.synchronize()
    o, f = out.view(batch * S, ent).cpu().long() & 0xffff, off.view(batch * S, NW + 1).cpu().long() & 0xffff
    mp = torch.full((NW * ws,), 255, dtype=torch.long)
    for sl in range(batch * S):
        for w in range(NW):
            a, e = int(f[sl, w]), int(f[sl, w + 1])
            mp[w * ws + o[sl, a:e]] = sl // S
    valid = torch.arange(pt.shape[1], device="cuda")[None, :] < (seq[:, None].long() + 1)
    ref = torch.full((NW * ws,), 255, dtype=torch.long)
    ref[pt[valid].long().cpu()] = torch.arange(batch, device="cuda")[:, None].expand_as(pt)[valid].cpu()
    print(f"B{batch}: rows {rows} WS {ws}; map ok {torch.equal(mp, ref)}", flush=True)
    for ch, nst in ((256, 2),):
        t0 = T(lambda: m.fz_run(kc, vc, ws, mg, sh, pt, seq, S, out, off, cnt, sink, E, per, 0, ch, nst))
        t1 = T(lambda: m.fz_run(kc, vc, ws, mg, sh, pt, seq, S, out, off, cnt, sink, E, per, 1, ch, nst))
        print(f"  CH {ch} NST {nst}: dense stream only {t0:.1f} us, + fused inversion {t1:.1f} us (exposed {t1 - t0:+.1f})", flush=True)
    ts = torch.zeros(132 * 8, dtype=torch.int64, device="cuda")
    for md, kk in ((1, kc), (3, kc), (1, kc[:0]), (3, kc[:0])):
      for _ in range(3):
        bench._FLUSH.zero_() if bench._FLUSH is not None else None; torch.cuda.synchronize()
        m.fz_run(kk, vc, ws, mg, sh, pt, seq, S, out, off, cnt, sink, ts, per, md, 256, 2); torch.cuda.synchronize()
      t = ts.view(132, 8)[:, :5].double().cpu(); t = (t - t[:, 0].min()) / 1000
      print(f"  MODE {md} stream {kk.shape[0] > 0} timeline (us; min/median/max over CTAs): " + "  ".join(f"{n} {t[:, i].min():.1f}/{t[:, i].median():.1f}/{t[:, i].max():.1f}" for i, n in enumerate(("start", "p1", "barrier", "map", "end"))), flush=True)
    del d, kc, vc; torch.cuda.empty_cache()
