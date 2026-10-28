# n6s4: H200 DIAG copy (NEVER submit): moe_twm K-split code compiled in (TWM_KSC) or not, layer big-M settings
# (HOT_M {256: 32}, TWF_SPLIT_M {256: 66} = the layer's T=256 tuner pick), layer v16/v32 routings. Variants (A B A B):
# k1 = KSC 1 (one twm build with the split code = the layer today), k2 = KSC 2 (dual build: calls without a split, i.e.
# v16, run the build without it; v32 runs the same split build as k1). Two libraries (prepare time: 300 s judge cap).
# -> /tmp/n6s4/dg/submission.py; read: grep "diag12 MEAN" <log>
import os
s = open('submission.py').read()
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
# compile time: the plain library (no DIAG-only tk2f families; these variants switch twm options only)
for x, y in (('        if TK2F_S6 or DIAG:  # n20', '        if TK2F_S6:  # n20'), ('        if DIAG:  # n4s5', '        if False:  # n4s5')):
    assert s.count(x) == 1, x
    s = s.replace(x, y)
# prepare-time cap (n6s3 HOTC DIAG was 302.7 s on the judge): time the big-M routings only
x = '        vs = DIAG_VARIANTS["dec" if T <= TK2_MAX else "big"]\n'
assert s.count(x) == 1
s = s.replace(x, '        if T <= TK2_MAX: continue\n' + x)
a = s.index('DIAG_VARIANTS = {'); b = s.index('\n}\n', a) + 3
L = '{"HOT_M": {256: 32}, "TWM_ORDER_M": {}, "HOT_NXB": 0'
V = [("k1", '{256: 66}', 1), ("k2", '{256: 66}', 2), ("k1b", '{256: 66}', 1), ("k2b", '{256: 66}', 2)]
s = s[:a] + ('DIAG_VARIANTS = {  # n6s4 DIAG copy\n    "dec": [("sq", {})],\n    "big": [' + ", ".join(
    f'("{n}", {L}, "TWF_SPLIT_M": {sp}, "TWM_KSC_FORCE": {k}}})' for n, sp, k in V) + '],\n}\n') + s[b:]
os.makedirs('/tmp/n6s4/dg', exist_ok=True)
open('/tmp/n6s4/dg/submission.py', 'w').write(s)
print('ok')
