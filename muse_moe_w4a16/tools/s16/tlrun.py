import sys, os
sys.path.insert(0, '/tmp/moe16')
import layer_ab as L, torch
sub = L.submission
for (l, sd), loc in sorted(L.R.items()):
    if l in sys.argv[1].split(",") and sd in (7, 11):
        data = L.mk(loc, sd)
        for o in [int(x) for x in sys.argv[2].split(",")]:
            sub._diag_tl(L.st, data, f"{l} s{sd}", o)
