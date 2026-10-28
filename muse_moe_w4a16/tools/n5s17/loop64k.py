"""n5s17: list hot loops (backward branches spanning > MIN bytes) that straddle a 64 KB boundary (function-relative).
H100 evidence (NOTES n5s17): a tk2f consumer loop crossing 0x10000 costs +4..12% (ncu stall_no_instruction 2-8x).
usage: python loop64k.py <file.so|cubin> [name-regex] [MIN=0x800]"""
import re, subprocess, sys
f = sys.argv[1]; rx = re.compile(sys.argv[2] if len(sys.argv) > 2 else "."); MIN = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x800
txt = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", f], capture_output=True, text=True).stdout
nf = nbad = 0
ALL = len(sys.argv) > 4 and sys.argv[4] == 'all'
VERB = len(sys.argv) > 4 and sys.argv[4] == 'v'
for p in re.split(r"\n\s*Function : ", txt)[1:]:
    name = p.split("\n", 1)[0].strip()
    if not rx.search(name): continue
    nf += 1; size = 0; bad = []; mma = []; loops = []
    for l in p.split("\n"):
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", l)
        if not m: continue
        a = int(m.group(1), 16); size = a
        if "MMA" in m.group(2): mma.append(a)
        if m.group(2).startswith("@") and " BRA " in m.group(2) + " ":  # predicated back-branch = loop bottom (unpredicated BRA = retry stubs)
            t = re.search(r"0x([0-9a-f]+)", m.group(2))
            if t:
                t = int(t.group(1), 16)
                if MIN < a - t < 0x40000: loops.append((t, a))
    loops = [(t, a, sum(t <= x <= a for x in mma)) for t, a in loops]
    # innermost loops holding MMAs: no other MMA loop nested inside (outer item/row loops are not the per-iteration hot path)
    inner = [(t, a, k) for t, a, k in loops if not any((t2, a2) != (t, a) and t <= t2 and a2 <= a and k2 for t2, a2, k2 in loops)]
    bad = [(t, a, k) for t, a, k in (loops if ALL else inner) if (t >> 16) != (a >> 16)]
    if not ALL: bad = [b for b in bad if b[2]]
    if VERB: print(name[:90], [(hex(t), hex(a), k) for t, a, k in inner if k])  # loops with (H|HG)MMA = consumer main loops (producer-only crossings: harmless)
    if bad:
        nbad += 1
        print(f"{name[:110]} size {size:#x}")
        for t, a, k in bad: print(f"    loop {t:#07x}-{a:#07x} ({(a - t) >> 10} KB, {k} mma)")
print(f"# {nf} functions, {nbad} with a back-branch across a 64 KB boundary")
