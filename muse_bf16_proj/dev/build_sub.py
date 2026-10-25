import os
d = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(d, "sub_template.py")).read()
for key, f in (("@@ROUTER@@", "router.cu"), ("@@GEMV@@", "gemv4.cu")):
    src = open(os.path.join(d, f)).read()
    assert "'''" not in src
    s = s.replace(key, src)
open(os.path.join(d, "..", "submission.py"), "w").write(s)
print("wrote submission.py")
