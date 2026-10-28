# n6s3: H200 DIAG copy (NEVER submit): TWM_EPI 0 vs 1 (A B A B) on the layer's v16/v32 routings with the layer's big-M
# settings, via _diag's DIAG_VARIANTS. -> /tmp/n6s3/dg2/submission.py; read: grep "diag12 MEAN" <log>
s = open('submission.py').read()
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
a = s.index('DIAG_VARIANTS = {'); b = s.index('\n}\n', a) + 3
L = '{"HOT_M": {256: 32}, "TWF_SPLIT_M": {256: 66}, "TWM_ORDER_M": {}, "HOT_NXB": 0'
s = s[:a] + ('DIAG_VARIANTS = {  # n6s3 DIAG copy\n    "dec": [("sq", {})],\n    "big": [' + ", ".join(
    f'("{n}", {L}, "TWM_EPI": {e}}})' for n, e in (("e0", 0), ("e1", 1), ("e0b", 0), ("e1b", 1))) + '],\n}\n') + s[b:]
open('/tmp/n6s3/dg2/submission.py', 'w').write(s)
print('ok')
