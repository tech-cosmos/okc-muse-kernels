"""python dev/s72/mkguest.py <guest submission.py> [driver.py] -> /tmp/pjg/submission.py (proj + GUEST; check only, never submit)."""
import base64, json, os, re, sys, zlib
d = os.path.dirname(os.path.abspath(__file__))
guest = open(sys.argv[1]).read()
driver = open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "moe_driver.py")).read()
if "moe_w4a16" in guest:  # moe: force DIAG on
    guest = re.sub(r"^DIAG = 0$", "DIAG = 1", guest, flags=re.M)
blob = base64.b64encode(zlib.compress(json.dumps({"sub": guest, "driver": driver}).encode(), 9)).decode()
s = open(os.path.join(d, "..", "..", "submission.py")).read()
assert s.count('\nGUEST = ""\n') == 1
# proj's own kernels off (torch fallback): the check then pays only the guest's compile (judge prepare limit 300 s)
imp = "    from torch.utils.cpp_extension import load_inline\n"
assert s.count(imp) == 1
s = s.replace(imp, "    def load_inline(*a, **k):\n        raise RuntimeError('GUEST build: proj kernels off')\n")
s = s.replace('\nGUEST = ""\n', f'\nGUEST = "{blob}"\n')
os.makedirs("/tmp/pjg", exist_ok=True)
open("/tmp/pjg/submission.py", "w").write(s)
print(f"wrote /tmp/pjg/submission.py ({len(s)} B, guest {len(guest)} B)")
