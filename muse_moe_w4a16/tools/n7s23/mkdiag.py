# n7s23: H200 DIAG copy (NEVER submit): TK2F_WAVE 0 vs 1 (check 87ee0bc0; earlier TK2F_XPF 0 vs 1, check 798fd574). Needs tools/n7s23/wave_xpf_dead.diff applied (reverted from main: both DEAD)
# A B A B on the layer's decode routings (d1/d16/d32/v1), one library (run-time l2h bit 14). Big-M skipped.
# -> /tmp/n7s23/submission.py; read: grep "diag12 MEAN" <log>
import os
s = open('submission.py').read()
for x, y in (('        if TK2F_S6 or DIAG:  # n20', '        if TK2F_S6:  # n20'), ('\nDIAG = 0\n', '\nDIAG = 1\n')):
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
V = [("w0", "{}"), ("w1", '{"TK2F_WAVE": 1}'), ("w0b", "{}"), ("w1b", '{"TK2F_WAVE": 1}')]
s = s[:a] + ('DIAG_VARIANTS = {  # n7s23 DIAG copy\n    "dec": [' + ", ".join(f'("{n}", {v})' for n, v in V) +
             '],\n    "big": [("h32", {})],\n}\n') + s[b:]
os.makedirs('/tmp/n7s23', exist_ok=True)
open('/tmp/n7s23/submission.py', 'w').write(s)
