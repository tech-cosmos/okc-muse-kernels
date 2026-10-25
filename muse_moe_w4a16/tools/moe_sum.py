import json, sys
d = json.load(open(sys.argv[1])); r = d['result']
print(d['id'], r.get('status'), r.get('passed'), r.get('speedup'), r.get('speedup_ci'), r.get('error'))
for ev in r.get('records', []):
    if ev.get('event') == 'benchmark':
        print(f"  bs {ev['case']['bs']:3d} ours {ev['median_ns']/1e3:7.1f} sglang {ev['baseline_ns']/1e3:7.1f} x{ev['speedup']}")
    elif ev.get('event') == 'test' and ev.get('status') != 'pass':
        print('  TEST', ev)
