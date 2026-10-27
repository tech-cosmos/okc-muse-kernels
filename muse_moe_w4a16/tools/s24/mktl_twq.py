# s24: build the twq(+HOTQ) per-CTA timeline DIAG copy -> /tmp/s24tl/submission.py (DIAG 1, TWM_TL 1, DIAG_TL on).
# okc check /tmp/s24tl/submission.py; grep diagtl <cwd>/profiles/submission.<job>.log. Note: _diag_tl's "TB/s" line uses
# twm's item byte classes (twq items are all 1.125 MB incl. scales: read the kinds/loss lines instead).
import os
os.makedirs("/tmp/s24tl", exist_ok=True)
s = open(os.path.join(os.path.dirname(__file__), "..", "..", "submission.py")).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a; s = s.replace(a, b)
rep("\nDIAG = 0\n", "\nDIAG = 1\n")
rep("#define TWM_TL 0", "#define TWM_TL 1")
i = s.index('DIAG_VARIANTS = {'); j = s.index('\n}\n', i) + 3
s = s[:i] + '''DIAG_VARIANTS = {  # s24 TL run: timing variants, then timelines (DIAG_TLV)
    "dec": [("sq", {})],
    "big": [("s66h32ne", {"RG": 0, "TWQ_NE": 1, "TWQ_M": {128: 536953348, 256: 4}}),
            ("s66ne", {"RG": 0, "TWQ_NE": 1, "TWQ_M": {128: 82436, 256: 4}})]
}
''' + s[j:]
i = s.index('\nDIAG_TL = ()'); j = s.index('\n', i + 1)
s = s[:i] + '\nDIAG_TL = (1,)' + s[j:]
i = s.index('\nDIAG_TLV = ['); j = s.index(']\n', i) + 2
s = s[:i] + '''
DIAG_TLV = [("s66h32ne", {"RG": 0, "TWQ_NE": 1, "TWQ_M": {128: 536953348}}), ("s66ne", {"RG": 0, "TWQ_NE": 1, "TWQ_M": {128: 82436}}),
            ("q4", {"RG": 0, "TWQ_NE": 0, "TWQ_M": {128: 4}})]
''' + s[j:]
open("/tmp/s24tl/submission.py", "w").write(s)
print("wrote /tmp/s24tl/submission.py")
