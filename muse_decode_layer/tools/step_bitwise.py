"""s63 race test for pre-wait loads (CSL_PW / vtk PW): per scored sliding shape and input D, run step A (= D moved back
by L tokens: it writes D's last L context rows) then step B (= D, caches not re-copied) as graph replays, (1) with a full
sync between them, twice (determinism), (2) back to back with no host sync (only the cold-L2 flush between, as the
judge), (3) B seeing the pre-A cache (stale: sensitivity). Bitwise compare of B's collected outputs.
python tools/step_bitwise.py [submission.py] [n] [filter] [force T:key=val,...]"""
import sys, torch
sys.path.insert(0, ".")
argv = sys.argv
src = open("tools/step_test.py").read().split("for load, c in bench.SCORED:")[0]
sys.argv = [argv[0], argv[1] if len(argv) > 1 else "submission.py", "0", argv[3] if len(argv) > 3 else ""] + argv[4:5]
exec(src)
n = int(argv[2]) if len(argv) > 2 else 6
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if c["layer"] != "sliding" or flt not in name:
        continue
    L = c["q_len"]
    data = bench.make_input(c, w, 7000)
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sub.kernel_fn(data, st)
        with torch.cuda.graph(gr, stream=s):
            out = sub.kernel_fn(data, st)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    cnt = dict(nondet=0, race=0, stale_seen=0)

    def run(d, mode):
        a = back(d, L)
        copy_into(data, a)
        kc0, vc0 = (data["k_cache"].clone(), data["v_cache"].clone()) if mode == "stale" else (None, None)
        torch.sum(flush.view(torch.int64), 0, out=sink)
        gr.replay()
        if mode != "nosync":
            torch.cuda.synchronize()
        copy_into(data, d, skip=("k_cache", "v_cache"))
        if mode == "stale":
            data["k_cache"].copy_(kc0); data["v_cache"].copy_(vc0)
        torch.sum(flush.view(torch.int64), 0, out=sink)
        gr.replay()
        torch.cuda.synchronize()
        return reference.collect(data, out).clone()

    for i in range(n):
        d = bench.make_input(c, w, 9001 + 31 * i)
        r1, r2, r3, r4 = run(d, "sync"), run(d, "sync"), run(d, "nosync"), run(d, "stale")
        cnt["nondet"] += not torch.equal(r1, r2)
        cnt["race"] += not torch.equal(r1, r3)
        cnt["stale_seen"] += not torch.equal(r1, r4)
        del d
    print(f"{name:16s} n {n} {cnt}", flush=True)
    del gr, data, out
    torch.cuda.empty_cache()
