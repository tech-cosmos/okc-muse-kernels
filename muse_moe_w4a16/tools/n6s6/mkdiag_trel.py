# n6s6: H200 DIAG copy (NEVER submit): TK2F_REL 0 vs 1 (tk2f/tk2h gemm13 publishes by a helper warp), A B A B, on the
# layer's decode routings (d1/d16/d32/v1), moe defaults (tk2f 204 / 84204 SQ) and the layer's
# T=32 tk2h W3 settings. Big-M routings skipped (prepare cap). -> /tmp/n6s6/dg/submission.py; read: grep "diag12 MEAN" <log>
import os
s = open('submission.py').read()
s = s.replace('\nDIAG = 0\n', '\nDIAG = 1\n', 1)
s = s.replace('\nTK2F_REL = 1 ', '\nTK2F_REL = 0 ', 1)
# compile time: the plain library + tk2h PF4 (TK2H_P4) only, no other DIAG-only tk2f families
for x, y in (('        if TK2F_S6 or DIAG:  # n20', '        if TK2F_S6:  # n20'),):
    assert s.count(x) == 1, x
    s = s.replace(x, y)
for x in ('            src = src.replace("#define TK2F_SQDEEP 0 ", "#define TK2F_SQDEEP 1 ")  # n5s15\n',
          '            src = src.replace("#define TK2F_PFL2K 0 ", "#define TK2F_PFL2K 1 ")  # n5s27\n',
          '            src = src.replace("#define TK2F_ORACLE 0 ", "#define TK2F_ORACLE 1 ")  # n5s16\n',
          '            src = src.replace("#define TK2X 0 ", "#define TK2X 2 ")  # n5s16\n',
          '            src = src.replace("#define TK2DT 0 ", "#define TK2DT 1 ")  # n5s18\n',
          '            src = src.replace("#define HOT_NX 0 ", "#define HOT_NX 1 ")  # n5s21\n'):
    assert s.count(x) == 1, x
    s = s.replace(x, '')
x = '        vs = DIAG_VARIANTS["dec" if T <= TK2_MAX else "big"]\n'
assert s.count(x) == 1
s = s.replace(x, '        if T > TK2_MAX: continue\n' + x)
a = s.index('DIAG_VARIANTS = {'); b = s.index('\n}\n', a) + 3
L = '"TK2H_M": {32: 3}'
V = [("m0", "{}"), ("m1", '{"TK2F_REL": 1}'), ("l0", "{" + L + "}"), ("l1", "{" + L + ', "TK2F_REL": 1}'),
     ("m0b", "{}"), ("m1b", '{"TK2F_REL": 1}'), ("l0b", "{" + L + "}"), ("l1b", "{" + L + ', "TK2F_REL": 1}')]
s = s[:a] + ('DIAG_VARIANTS = {  # n6s6 DIAG copy\n    "dec": [' + ", ".join(f'("{n}", {v})' for n, v in V) +
             '],\n    "big": [("h32", {})],\n}\n') + s[b:]
os.makedirs('/tmp/n6s6/dg', exist_ok=True)
open('/tmp/n6s6/dg/submission.py', 'w').write(s)
print('ok')
