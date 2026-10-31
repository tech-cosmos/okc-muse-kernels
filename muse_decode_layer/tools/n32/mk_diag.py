# DIAG ONLY (never submit): SM-clock probe in the layer tuner, for proj n24's T=256 full confirm-loop question.
import sys
src = open(sys.argv[1]).read()
anchor = "            state[\"tune\"][key] = cands[best]\n"
assert src.count(anchor) == 1
diag = anchor + '''            if (tokens, sliding) in ((256, False), (256, True), (16, True)):
                def _clk():
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); torch.cuda._sleep(400000); b.record(); b.synchronize()
                    return 4e5 / (a.elapsed_time(b) * 1e3)
                import time as _t
                _t.sleep(0.3)
                idle = sorted(_clk() for _ in range(9))[4]
                tt, cc = [], []
                t0 = _t.perf_counter()
                for r in range(1500):
                    flush.zero_()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); graphs[0].replay(); b.record(); b.synchronize()
                    tt.append(a.elapsed_time(b) * 1e3)
                    if r % 5 == 0:
                        cc.append(_clk())
                wall = _t.perf_counter() - t0
                n6, m6 = len(tt) // 6, len(cc) // 6
                md = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
                print(f"DIAGCLK T={tokens} {'sl' if sliding else 'full'} idle {idle:.0f}MHz wall {wall:.2f}s | us per sixth "
                      + " ".join(f"{md(tt[i*n6:(i+1)*n6]):.1f}" for i in range(6)) + " | MHz per sixth "
                      + " ".join(f"{md(cc[i*m6:(i+1)*m6]):.0f}" for i in range(6)) + f" | min {min(cc):.0f}", flush=True)
                _t.sleep(0.5)
                idle2 = sorted(_clk() for _ in range(9))[4]
                tt2 = []
                for r in range(60):
                    flush.zero_()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); graphs[0].replay(); b.record(); b.synchronize()
                    tt2.append(a.elapsed_time(b) * 1e3)
                print(f"DIAGCLK T={tokens} {'sl' if sliding else 'full'} after 0.5s rest: idle {idle2:.0f}MHz, 60 replays {md(tt2):.1f}us", flush=True)
'''
src = src.replace(anchor, diag)
open(sys.argv[2], "w").write(src)
