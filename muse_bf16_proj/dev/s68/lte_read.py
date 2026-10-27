"""Readout of a judge log's `diaglte` lines (s68 _diag_lte): per shape F.linear, best new config, best lt_cfg.txt config
(graph-replay us incl. the ~4.6 us 1-kernel graph cost, cold L2). Usage: python lte_read.py <judge .log> [more logs]"""
import ast, os, re, sys

LT = ast.literal_eval(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lt_cfg.txt")).read())

rows = {}
for path in sys.argv[1:]:
    for line in open(path):
        m = re.match(r"diaglte (\w+) m(\d+): (\d+) cfgs, [\d.]+ s; F ([\d.]+) \| new (.*?) \| lt_cfg(.*)$", line.strip())
        if not m:
            continue
        layer, mm, ncfg, f, new, old = m.groups()
        new = [(ast.literal_eval(c), float(t.split("/")[0])) for c, t in re.findall(r"(\([^)]*\)):([\d./]+)", new)]
        old = [(ast.literal_eval(c), float(t)) for c, t in re.findall(r"(\([^)]*\)):([\d.]+)", old)]
        rows.setdefault((layer, int(mm)), []).append((float(f), new, old))
for (layer, m), runs in sorted(rows.items()):
    for f, new, old in runs:
        mine = set(LT.get((layer, m), []))
        old = old + [r for r in new if r[0] in mine]
        new = [r for r in new if r[0] not in mine]
        bn = min(new, key=lambda r: r[1]) if new else (None, 9e9)
        bo = min(old, key=lambda r: r[1]) if old else (None, 9e9)
        print(f"{layer:9s} m{m:<4d} F {f:6.2f} | best new {bn[1]:6.2f} {bn[0]} | best lt_cfg {bo[1]:6.2f} {bo[0]} | "
              f"new - lt_cfg {bn[1] - bo[1]:+.2f}")
