import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bndc as B
import submission as L
B.lib()
def mk(bg, cs):
    return lambda a, s, al, be, g1, g2, so, h: B.lib().bndc(a, s, al, be, g2, g1, so, h, L.POST_NORM_EPS, L.RMS_EPS, cs, bg)
VARIANTS = {f"c{cs}" + "g" * bg: mk(bg, cs) for cs in (1, 2, 4, 8) for bg in (False, True)}
def mp(bg, r):
    return lambda a, s, al, be, g1, g2, so, h: B.lib().bndp(a, s, al, be, g2, g1, so, h, L.POST_NORM_EPS, L.RMS_EPS, r, bg)
VARIANTS.update({f"p{r}" + "g" * bg: mp(bg, r) for r in (1, 2, 3, 4) for bg in (False, True)})
