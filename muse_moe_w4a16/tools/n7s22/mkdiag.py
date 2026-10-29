# n7s22: H200 DIAG copy (NEVER submit): per-CTA timeline (TK2F_IDH 1, T <= 8) + TK2F_IDH 0 vs 1 A B A B on the layer's
# decode routings (d1/d16/d32/v1). Big-M skipped. -> /tmp/n7s22/submission.py; read: grep "diagtl\|diag12 MEAN" <log>
import os, subprocess, sys
s = open('submission.py').read()
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
V = [("i1", "{}"), ("i0", '{"TK2F_IDH": 0}'), ("i1b", "{}"), ("i0b", '{"TK2F_IDH": 0}')]
s = s[:a] + ('DIAG_VARIANTS = {  # n7s22 DIAG copy\n    "dec": [' + ", ".join(f'("{n}", {v})' for n, v in V) +
             '],\n    "big": [("h32", {})],\n}\n') + s[b:]
os.makedirs('/tmp/n7s22', exist_ok=True)
open('/tmp/n7s22/pre.py', 'w').write(s)
# arg 'ab': + the IDH A/B (2 libraries: local prepare 235 s, too close to the judge's 300 s cap); default timeline only
AB = len(sys.argv) > 1 and sys.argv[1] == 'ab'
env = dict(os.environ, SRC='/tmp/n7s22/pre.py')
if AB: env['BOTH'] = '1'
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), 'mktl.py'), '8' if AB else '32'], env=env)
