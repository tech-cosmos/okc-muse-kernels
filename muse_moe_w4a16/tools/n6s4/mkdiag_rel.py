# n6s4: H200 DIAG copy (NEVER submit): TWM_REL 0 vs 1 (moe_twm gemm13 release by helper warp 9), A B A B, both with the
# dual K-split build (TWM_KSC_FORCE 2) and the layer's big-M settings (HOT_M {256: 32}, TWF_SPLIT_M {256: 66}), layer
# v16/v32 routings. -> /tmp/n6s4/dgr/submission.py; read: grep "diag12 MEAN" <log>
import os
s = open('submission.py').read()
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
a = s.index('DIAG_VARIANTS = {'); b = s.index('\n}\n', a) + 3
L = '{"HOT_M": {256: 32}, "TWM_ORDER_M": {}, "HOT_NXB": 0, "TWF_SPLIT_M": {256: 66}, "TWM_KSC_FORCE": 2'
V = [("r0", 0), ("r1", 1), ("r0b", 0), ("r1b", 1)]
s = s[:a] + ('DIAG_VARIANTS = {  # n6s4 DIAG copy\n    "dec": [("sq", {})],\n    "big": [' + ", ".join(
    f'("{n}", {L}, "TWM_REL": {r}}})' for n, r in V) + '],\n}\n') + s[b:]
os.makedirs('/tmp/n6s4/dgr', exist_ok=True)
open('/tmp/n6s4/dgr/submission.py', 'w').write(s)
print('ok')
