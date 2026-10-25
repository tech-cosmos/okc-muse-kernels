"""Does an L2 bulk prefetch of the NEXT weight, run concurrently on a side stream during a big GEMV, make
the pair faster? A = 64 MB GEMV (m=1), B = GEMV on a second weight of MB megabytes."""
import sys, fcntl, statistics, torch, triton, triton.language as tl, torch.nn.functional as F
sys.path.insert(0, "tools")
from pfprobe import pf, t, gemv, x  # noqa
dev = "cuda"
for mb in (8, 16, 24):
    wb = torch.randn(mb * 64, 8192, device=dev).bfloat16()
    gb = lambda: F.linear(x, wb)
    side = torch.cuda.Stream()
    def par(P, ch=1024, nw=1):
        def f():
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                pf[(P,)](wb, ch, CHE=ch // 2, NT=nw * 32, num_warps=nw)
            gemv()
            cur.wait_stream(side)
            gb()
        return f
    P = (mb << 20) // (1024 * 32)
    print(f"B {mb}MB: A;B {t([gemv, gb]):6.2f}  B alone {t([gb]):6.2f}  A||pf(B);B {t([par(P)]):6.2f}  "
          f"half pf {t([par(P // 2)]):6.2f}")
