"""python dev/s72/mkguest.py moe <moe submission.py>          (driver dev/s72/moe_driver.py)
   python dev/s72/mkguest.py layer <r-layer muse_decode_layer dir> (submission.py + nvfp4/checks/reference; driver layer_driver.py)
-> /tmp/pjg/submission.py = proj + GUEST (proj kernels off). Check only, NEVER submit."""
import base64, json, os, re, sys, zlib
d = os.path.dirname(os.path.abspath(__file__))
kind, path = sys.argv[1], sys.argv[2]
mods, order = {}, []
if kind == "moe":
    guest = open(path).read()
    guest = re.sub(r"^DIAG = 0$", "DIAG = 1", guest, flags=re.M)  # moe: force DIAG on
elif kind == "layer":  # the layer's own DIAG switches must already be set in its submission.py (e.g. _moe_DIAG = 1)
    guest = open(os.path.join(path, "submission.py")).read()
    order = ["nvfp4", "checks", "reference"]
    mods = {m: open(os.path.join(path, m + ".py")).read() for m in order}
    for m in order:  # `from <helper> import a, b` -> names from the helper's namespace (no sys.modules: the judge rejects edits)
        mods[m] = re.sub(r"^from (" + "|".join(order) + r") import (.+)$",
                         lambda mm: "; ".join(f"{n.strip()} = _guest_mods['{mm.group(1)}']['{n.strip()}']"
                                              for n in mm.group(2).split(",")), mods[m], flags=re.M)
else:
    sys.exit("kind = moe | layer")
driver = open(os.path.join(d, f"{kind}_driver.py")).read()
blob = base64.b64encode(zlib.compress(json.dumps({"sub": guest, "driver": driver, "mods": mods, "order": order}).encode(), 9)).decode()
s = open(os.path.join(d, "..", "..", "submission.py")).read()
assert s.count('\nGUEST = ""\n') == 1
s = s.replace('\nGUEST = ""\n', f'\nGUEST = "{blob}"\n')
# proj's own kernels off (torch fallback): the check then pays only the guest's compile (judge prepare limit 300 s)
imp = "    from torch.utils.cpp_extension import load_inline\n"
assert s.count(imp) == 1
s = s.replace(imp, "    def load_inline(*a, **k):\n        raise RuntimeError('GUEST build: proj kernels off')\n")
os.makedirs("/tmp/pjg", exist_ok=True)
open("/tmp/pjg/submission.py", "w").write(s)
print(f"wrote /tmp/pjg/submission.py ({len(s)} B, guest {len(guest)} B, mods {list(mods)})")
