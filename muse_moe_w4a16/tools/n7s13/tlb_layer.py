# Usage: sed "s/^#define TWM_TL 0$/#define TWM_TL 1/" submission.py > /tmp/x/submission.py; cp tools/n7s13/tlb_layer.py /tmp/x/;
#   local: (cd muse_moe_w4a16; python /tmp/x/tlb_layer.py)   H200: python tools/h200.py muse_moe_w4a16 tl=/tmp/x --cmd "python tlb_layer.py"
# n7s13: R1 gate on the LAYER's v16/v32 routings (tools/data/routes.pt, as tools/n6s3/ab2.py). Same as n7s1/tlb_layer.py.
# n7s1: R1 gate on board big-M inputs. Per-CTA twm timeline (TWM_TL build), cold L2 before the stamped call.
import sys, torch, statistics as S
sys.path.append(".")
import reference, submission as m, bench
W, st = bench.state()
fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sk = torch.empty(1, dtype=torch.int64, device="cuda")
R = torch.load('tools/data/routes.pt')
def mk(loc, seed):
    T = loc.shape[0]; g = torch.Generator(device='cuda').manual_seed(seed)
    ids = torch.where(loc.cuda() >= 0, loc.cuda().long() + st["start"], (st["start"] + st["count"] + 5) % 256).int()
    tw = torch.rand(T, 8, device='cuda', generator=g); tw = tw / tw.sum(-1, keepdim=True)
    return {"hidden_states": torch.randn(T, 4096, device='cuda', generator=g).bfloat16(), "topk_ids": ids, "topk_weights": tw,
            "config": {"bs": T, "d_hidden": 4096, "local_start": st["start"], "local_count": st["count"]}}
for (lab, seed), loc in sorted(R.items()):
    if loc.shape[0] > 32:
        bs = loc.shape[0]; d = mk(loc, seed)
        full = bench.time_graph(lambda: m.kernel_fn(d, st), 20) - bench.empty_cost()
        m.DIAG_TL = ("x",)
        old = dict(m.TWM_ORDER_M); m.TWM_ORDER_M = dict(old); m.TWM_ORDER_M[bs] = m.TWM_ORDER_M.get(bs, m.TWM_ORDER) | 2
        for _ in range(3):
            torch.sum(fl.view(torch.int64), 0, out=sk); m.kernel_fn(d, st)
        torch.cuda.synchronize()
        m.DIAG_TL = (); m.TWM_ORDER_M = old
        part, meta = st.pop("_tl")
        tl = part[66 * 32768:].view(torch.int64)[:132 * 64].view(132, 64).cpu()
        act = [c for c in range(132) if tl[c, 63].item() > 0]
        t0 = min(tl[c, 0].item() for c in act)
        u = lambda v: (v - t0) / 1e3
        T = max(u(tl[c, 63].item()) for c in act)
        busy = ramp = gaps = drain = 0.0; ni = 0; kinds = {}
        for c in act:
            n = min(tl[c, 1].item(), 20); pe = None
            for k in range(n):
                a, b, dd = u(tl[c, 2 + 3 * k].item()), u(tl[c, 3 + 3 * k].item()), tl[c, 4 + 3 * k].item()
                busy += b - a; ni += 1
                key = (dd & 1, (dd >> 1) & 3, ((dd >> 8) & 255) + 7 >> 3 << 3)
                kinds.setdefault(key, []).append(b - a)
                if pe is None: ramp += a
                else: gaps += a - pe
                pe = b
            drain += T - (pe if pe is not None else 0)
        G = len(act)
        print(f"{lab} s{seed} (T{bs}): full call {full:.1f} us | twm span {T:.1f} (call - span {full - T:.1f}) | G {G} items {ni} | per CTA: "
              f"busy {busy/G:.1f} ramp {ramp/G:.1f} gaps {gaps/G:.1f} drain {drain/G:.1f} | idle {100*(1-busy/G/T):.1f}%", flush=True)
        print("   kinds (g2 cl N: n mean us): " + "; ".join(f"{k[0]}{k[1]} N{k[2]}: {len(v)} {S.mean(v):.1f}" for k, v in sorted(kinds.items())), flush=True)
