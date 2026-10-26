import os
d = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(d, "sub_template.py")).read()
for key, f in (("@@ROUTER@@", "router2.cu"), ("@@GEMV7@@", "gemv7.cu"), ("@@GEMV10@@", "gemv10.cu"), ("@@GEMV12@@", "gemv12.cu"), ("@@GEMV13@@", "gemv13.cu"), ("@@GEMV14@@", "gemv14.cu"), ("@@GEMV15@@", "gemv15.cu"), ("@@GEMV16@@", "gemv16.cu"), ("@@GEMV17@@", "gemv17.cu"), ("@@ROUTER3@@", "router3.cu"), ("@@GEMV19@@", "gemv19.cu"), ("@@LT@@", "lt.cu")):
    src = open(os.path.join(d, f)).read()
    assert "'''" not in src
    s = s.replace(key, src)
s = s.replace("@@LTCFG@@", open(os.path.join(d, "lt_cfg.txt")).read().strip())
open(os.path.join(d, "..", "submission.py"), "w").write(s)
print("wrote submission.py")
