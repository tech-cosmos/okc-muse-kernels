# n6s3: H200 DIAG copy (NEVER submit): TWM_HOTC 1 vs 0 (A B A B; TWM_EPI 1 both) on the layer's v16/v32 routings, layer big-M
# settings. -> /tmp/n6s3/dg3/submission.py; read: grep "diag12 MEAN" <log>
import os
s = open('submission.py').read()
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
a = s.index('DIAG_VARIANTS = {'); b = s.index('\n}\n', a) + 3
L = '{"HOT_M": {256: 32}, "TWF_SPLIT_M": {256: 66}, "TWM_ORDER_M": {}, "HOT_NXB": 0'
s = s[:a] + ('DIAG_VARIANTS = {  # n6s3 DIAG copy\n    "dec": [("sq", {})],\n    "big": [' + ", ".join(
    f'("{n}", {L}, "TWM_HOTC_FORCE": {e}}})' for n, e in (("h1", 1), ("h0", 0), ("h1b", 1), ("h0b", 0))) + '],\n}\n') + s[b:]
os.makedirs('/tmp/n6s3/dg3', exist_ok=True)
open('/tmp/n6s3/dg3/submission.py', 'w').write(s)
print('ok')
