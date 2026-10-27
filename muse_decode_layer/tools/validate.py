"""One-command validation of a layer bundle (proj s75, lane (b) per lead 2026-10-27 08:55Z).
python tools/validate.py [submission.py] [--quick] [--skip=step,...] [--only=step,...]
Runs, each as its own process (logs in /tmp/validate/<tag>/<step>.log), then prints one summary line per step and a
NOTES-ready line:
  test     bench.py test (the judge's 13 tests)
  scored   tools/scored_test.py all scored shapes x 2 seeds (1 with --quick)
  opt      tools/opt_test.py every TUNE option forced on TUNE_BASE x 1 seed
  tk2f     tools/tk2f_stress.py 300 (75 --quick): tk2f vs fallback, bitwise
  replay   tools/replay_test.py 20 inputs, router seeds 11,23 (11 --quick)
  route    tools/route_judge.py 20 inputs, router seed 11
  fuzz     tools/stream_fuzz.py (proj s83): every TUNE set with a spin kernel injected at each side/hi-stream fork (side late /
           main late, eager + CUDA graph), reps 2 per forking set (1 and side modes only with --quick)
  prep     tools/prep_time.py with fresh TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR (cold; judge limit 300 s, keep <= 200 s)
Correctness steps take no GPU lock (they are not timed); prep_time is wall-clock and is NOT locked either: its number
is pessimistic when other agents load the GPU. Exit code 1 if any step fails."""
import os, re, subprocess, sys, time, shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
sub = os.path.abspath(args[0] if args else os.path.join(HERE, "submission.py"))
quick = "--quick" in flags
tag = re.sub(r"\W", "_", os.path.relpath(sub, HERE)) + time.strftime("_%H%M%S")
logd = f"/tmp/validate/{tag}"
os.makedirs(logd, exist_ok=True)
PY = sys.executable


def count(pat, txt):
    return len(re.findall(pat, txt, re.M))


def p_test(t):
    return count(r"^FAIL", t), count(r"^PASS", t) + count(r"^FAIL", t), ""


def p_bad(t):  # scored/opt/tk2f: final "bad N [of M]"
    m = re.findall(r"^bad (\d+)(?: of (\d+))?", t, re.M)
    if not m:
        return None, None, "no summary"
    b, n = m[-1]
    n = n or str(count(r"^(PASS|FAIL)", t))
    return int(b), int(n), ""


def p_rs(t):  # replay/route: "rs <seed> <shape> fails a/b"
    m = re.findall(r"fails (\d+)/(\d+)", t)
    if not m:
        return None, None, "no summary"
    extra = ""
    g = [float(x) for x in re.findall(r"ours gap ([\d.]+)", t)]
    d = [float(x) for x in re.findall(r"drop-lead ([\d.]+) \|", t)]
    if g:
        extra = f"worst gap {max(g):.4f} drop-lead {max(d):.4f}"
    return sum(int(a) for a, _ in m), sum(int(b) for _, b in m), extra


def p_prep(t):
    m = re.findall(r"^prepare ([\d.]+)s total", t, re.M)
    if not m:
        return None, None, "no total"
    s = float(m[-1])
    return (1 if s > 200 else 0), 1, f"{s:.1f} s"


STEPS = [
    ("test", [PY, "bench.py", "test", sub], {}, p_test),
    ("scored", [PY, "tools/scored_test.py", sub, "", "1" if quick else "2"], {}, p_bad),
    ("opt", [PY, "tools/opt_test.py", sub, "1"], {}, p_bad),
    ("tk2f", [PY, "tools/tk2f_stress.py", "75" if quick else "300"], {"SUB": sub}, p_bad),
    ("replay", [PY, "tools/replay_test.py", sub, "20", "11" if quick else "11,23"], {}, p_rs),
    ("route", [PY, "tools/route_judge.py", sub, "20", "11"], {}, p_rs),
    ("fuzz", [PY, "tools/stream_fuzz.py", sub, "side,gside" if quick else "side,main,gside,gmain", "400000", "1", "",
              "1" if quick else "2"], {}, p_bad),
    ("prep", [PY, "tools/prep_time.py", sub], "COLD", p_prep),
]
only = set(flags["--only"].split(",")) if "--only" in flags else None
skip = set(flags.get("--skip", "").split(","))
res, bad_any = [], False
print(f"validate {sub} -> logs {logd}", flush=True)
for name, cmd, env, parse in STEPS:
    if (only and name not in only) or name in skip:
        continue
    e = dict(os.environ)
    if env == "COLD":
        cd = f"/tmp/validate/cold_{tag}"
        shutil.rmtree(cd, ignore_errors=True)
        e.update(TORCH_EXTENSIONS_DIR=f"{cd}/te", TRITON_CACHE_DIR=f"{cd}/tr")
    else:
        e.update(env)
    t0 = time.time()
    with open(f"{logd}/{name}.log", "w") as f:
        rc = subprocess.run(cmd, cwd=HERE, env=e, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = open(f"{logd}/{name}.log").read()
    b, n, extra = parse(txt)
    ok = b == 0 and rc == 0
    if rc != 0:
        extra = (extra + f" rc={rc}").strip()
    if env == "COLD":
        shutil.rmtree(f"/tmp/validate/cold_{tag}", ignore_errors=True)
    bad_any |= not ok
    line = f"{name:7s} {'OK  ' if ok else 'FAIL'} {b}/{n} bad  {extra}  ({time.time()-t0:.0f} s)"
    res.append((name, ok, b, n, extra))
    print(line, flush=True)
    if not ok:
        print("   tail:", *txt.strip().splitlines()[-4:], sep="\n     ", flush=True)
short = {"test": "bench test", "scored": "scored", "opt": "opt_test", "tk2f": "tk2f_stress", "replay": "replay",
         "route": "route_judge", "fuzz": "stream_fuzz", "prep": "cold prepare"}
parts = []
for name, ok, b, n, extra in res:
    if name == "test":
        parts.append(f"bench test {n - b}/{n}")
    elif name == "prep":
        parts.append(f"cold prepare {extra}")
    else:
        parts.append(f"{short[name]} {b}/{n}" + (f" ({extra})" if extra and name == "route" else ""))
print("NOTES:", ("ALL OK: " if not bad_any else "FAILURES: ") + ", ".join(parts), flush=True)
sys.exit(1 if bad_any else 0)
