"""n27: record every load_inline source a submission compiles (prepare + 13 tests + the 12 scored shapes), or compare two
records modulo C++ // comments (build.py strip_cuda_comments must change nothing else, also after runtime text patches).
  python tools/n27/srccmp.py rec <submission.py> <out.json>
  python tools/n27/srccmp.py cmp <a.json> <b.json>"""
import json, re, sys


def norm(s):
    out = []
    for ln in s.split("\n"):
        i, inq = 0, False
        while i < len(ln) and (inq or ln[i:i + 2] != "//"):
            inq ^= ln[i] == '"' and (i == 0 or ln[i - 1] != "\\")
            i += 1
        ln = ln[:i].rstrip()
        if ln.strip():
            out.append(ln)
    return "\n".join(out)


def flat(v):
    return "\n".join(v) if isinstance(v, (list, tuple)) else (v or "")


if sys.argv[1] == "rec":
    sys.path.insert(0, ".")
    import torch.utils.cpp_extension as ce
    rec, real = [], ce.load_inline

    def li(*a, **kw):
        rec.append({k: (flat(v) if k in ("cpp_sources", "cuda_sources") else repr(v)) for k, v in kw.items() if k != "name"})
        return real(*a, **kw)
    ce.load_inline = li
    import bench, torch
    sub = bench.load_sub(sys.argv[2])
    bench.test(sub)
    w, st = bench.weights_state(sub)
    for i, (load, c) in enumerate(bench.SCORED):
        sub.kernel_fn(bench.make_input(c, w, 1 + i), st)
        torch.cuda.synchronize()
    json.dump(rec, open(sys.argv[3], "w"))
    print("recorded", len(rec), "load_inline calls")
else:
    a, b = (json.load(open(f)) for f in sys.argv[2:4])
    key = lambda r: json.dumps({k: (norm(v) if k.endswith("sources") else v) for k, v in sorted(r.items())}, sort_keys=True)
    ka, kb = sorted(map(key, a)), sorted(map(key, b))
    print(len(a), len(b), "calls; raw bytes", sum(len(r.get("cuda_sources", "")) for r in a), sum(len(r.get("cuda_sources", "")) for r in b))
    print("IDENTICAL modulo comments" if ka == kb else "DIFFERENT")
    for x in set(ka) ^ set(kb):
        print("only in", "a" if x in ka else "b", x[:300])
