# per-stage consumer timeline of moe_twm items: wait-on-full cycles vs work cycles (consumer warp 0 of each CTA)
import sys, os, torch, statistics as S
sys.path.insert(0, '.')
import layer_ab as L
sub = L.submission
lib = L.st["lib"]
buf = torch.empty(132 * 1024, dtype=torch.int64)
def run(name, loc, warm):
    data = L.mk(loc, 1)
    for _ in range(3): sub.kernel_fn(data, L.st)
    torch.cuda.synchronize()
    if not warm: L.torch.sum(L.flush.view(torch.int64), 0, out=L.sink)
    lib.moe_stl_clear(); sub.kernel_fn(data, L.st); torch.cuda.synchronize()
    lib.moe_stl_get(buf)
    b = buf.view(132, 512, 2)
    G = int(os.environ.get("MOE_GRID", "132"))
    W, C = [], []
    for c in range(G):
        for k in range(512):
            w, cc = b[c, k, 0].item(), b[c, k, 1].item()
            if w == 0 and cc == 0: break
            if k % 64: W.append(w); C.append(cc)   # skip item-boundary stages
    tw, tc = sum(W), sum(C)
    print(f"{name}: stages {len(W)}  wait med {S.median(W)} mean {tw/len(W):.0f}  work med {S.median(C)} mean {tc/len(C):.0f} cyc"
          f"  wait share {tw/(tw+tc):.2f}", flush=True)
T = 64
loc = torch.full((T, 8), -1, dtype=torch.int32); loc[:, 0] = 0
run("1 expert x64 tok (RT2 N64) warm", loc, True)
loc = torch.full((T, 8), -1, dtype=torch.int32); loc[:32, 0] = 0
run("1 expert x32 tok (RT4 N32) warm", loc, True)
loc = torch.full((T, 8), -1, dtype=torch.int32)
for e in range(8): loc[e * 8:(e + 1) * 8, 0] = e
run("8 experts x8 tok (RT4 N8) warm", loc, True)
for (l, sd), loc in sorted(L.R.items()):
    if l in ("verify 16", "verify 32") and sd == 7: run(f"{l} s7 cold", loc, False)
