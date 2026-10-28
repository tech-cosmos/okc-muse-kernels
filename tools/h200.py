#!/usr/bin/env python3
"""A/B timing on a Modal H200 in the judge's image, to pre-screen candidates before spending a judge check.

  python tools/h200.py <problem_dir> base=<file|dir> cand=<file|dir> [--cmd 'python bench.py bench'] [--rounds 1]

Each variant is the problem dir with submission.py replaced (a file) or overlaid (a dir); all variants run in ONE
container on the same GPU, in order A B (rounds 2: A B B A). Modal's absolute times differ from the judge's by a
steady 4-17% per shape (s: 2026-10-28 test: faster at bs16-256, slower at bs1), so compare variants within one run,
never against judge numbers.

Budget: a shared daily cap in GPU-minutes (H200_DAILY_MIN, default 60 = ~$4.5/UTC day) kept in ~/.h200_ledger.json
under a file lock; each run reserves its timeout up front and is charged its real wall time. Needs a Modal token
(~/.modal.toml), which the human installs; agents never create or change credentials.
"""
import fcntl, io, json, os, sys, tarfile, time, datetime, pathlib

LEDGER = pathlib.Path.home() / ".h200_ledger.json"
DAILY_MIN = float(os.environ.get("H200_DAILY_MIN", "60"))
MAX_RUN_MIN = 25
SKIP = {"dev", "profiles", "__pycache__", ".git"}


def ledger(update):
    LEDGER.touch(exist_ok=True)
    with open(LEDGER, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.loads(f.read() or "{}")
        out = update(d)
        f.seek(0); f.truncate(); f.write(json.dumps(d, indent=1))
        return out


def today():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def tar_variant(problem, override):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(problem, arcname=".", filter=lambda ti: None if set(pathlib.Path(ti.name).parts) & SKIP else ti)
        o = pathlib.Path(override)
        if o.is_file():
            t.add(o, arcname="./submission.py")
        else:
            t.add(o, arcname=".", filter=lambda ti: None if set(pathlib.Path(ti.name).parts) & SKIP else ti)
    return buf.getvalue()


def main():
    args = sys.argv[1:]
    cmd, rounds = "python bench.py bench", 1
    if "--cmd" in args:
        i = args.index("--cmd"); cmd = args[i + 1]; del args[i:i + 2]
    if "--rounds" in args:
        i = args.index("--rounds"); rounds = int(args[i + 1]); del args[i:i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    problem, variants = args[0], [a.split("=", 1) for a in args[1:]]
    order = [v for v in variants] if rounds == 1 else variants + variants[::-1]
    est_min = min(MAX_RUN_MIN, 3 + 4 * len(order))  # container start + ~2 min prepare + bench per variant

    def reserve(d):
        used = d.get(today(), 0.0)
        if used + est_min > DAILY_MIN:
            return None
        d[today()] = used + est_min
        return used
    used = ledger(reserve)
    if used is None:
        sys.exit(f"h200: daily cap reached ({ledger(lambda d: d.get(today(), 0))} of {DAILY_MIN} GPU-min used today); "
                 "use a judge check or wait for 00:00 UTC")
    print(f"h200: reserved {est_min} min ({used:.0f} of {DAILY_MIN:.0f} used today before this run)", flush=True)

    import modal
    image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open")
    app = modal.App("okc-h200-ab", image=image)

    @app.function(gpu="H200", timeout=est_min * 60, serialized=True)
    def run(blobs, cmd):
        import subprocess, tarfile, io, os
        out = [subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                              capture_output=True, text=True).stdout]
        for name, blob in blobs:
            w = f"/work/{name}-{len(out)}"
            os.makedirs(w)
            tarfile.open(fileobj=io.BytesIO(blob)).extractall(w)
            r = subprocess.run(cmd, shell=True, cwd=w, capture_output=True, text=True)
            out.append(f"===== {name} (rc {r.returncode})\n{r.stdout[-5000:]}\n{r.stderr[-1500:] if r.returncode else ''}")
        return "\n".join(out)

    blobs = [(n, tar_variant(problem, p)) for n, p in order]
    t0 = time.time()
    try:
        with modal.enable_output() if os.environ.get("H200_VERBOSE") else open(os.devnull) as _:
            with app.run():
                print(run.remote(blobs, cmd))
    finally:
        mins = (time.time() - t0) / 60
        ledger(lambda d: d.__setitem__(today(), d.get(today(), 0) - est_min + mins))
        print(f"h200: charged {mins:.1f} GPU-min; {ledger(lambda d: d.get(today(), 0)):.0f} of {DAILY_MIN:.0f} used today")


if __name__ == "__main__":
    main()
