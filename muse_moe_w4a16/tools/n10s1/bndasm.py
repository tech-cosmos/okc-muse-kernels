import sys, torch, re
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
A = torch.load("/tmp/moeend/bnd_args.pt", weights_only=False)
for nm, (g, a, k) in A.items():
    ck = sub._boundary[g](*a, **k)
    sass = ck.asm.get("sass") or ""
    ptx = ck.asm["ptx"]
    L = ptx.split("\n")
    wi = [i for i, l in enumerate(L) if "griddepcontrol.wait" in l]
    ld = [i for i, l in enumerate(L) if re.search(r"\bld\.global", l)]
    print(nm, "ptx wait at", wi, "ld.global before:", sum(i < wi[0] for i in ld), "after:", sum(i > wi[0] for i in ld),
          "bar.sync:", sum("bar.sync" in l or "barrier.sync" in l for l in L), "regs", ck.n_regs, "spills", ck.n_spills)
    if sass:
        S = sass.split("\n")
        wi = [i for i, l in enumerate(S) if "ACQBULK" in l or "griddepcontrol" in l.lower() or "DEPBAR" in l and False]
        print("  sass lines", len(S))
    open(f"/tmp/moeend/bnd_{nm.split()[0]}{nm.split()[1]}.ptx", "w").write(ptx)
