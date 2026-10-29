# n7s41: reads the n7s22 H200 DIAG 81391142 (stamped tk2f, layer routings): /tmp/n7s22/run/check.json + /tmp/n7s22/run/submission.py (routings); run from the dir holding routes.py
import re, zlib, base64, ast, numpy as np
s=open('/tmp/n7s22/run/submission.py').read()
hdr=ast.literal_eval(re.search(r'^_DIAG_HDR = (.*)$',s,re.M).group(1))
raw=zlib.decompress(base64.b64decode(re.search(r'^_DIAG_R = "(.*)"$',s,re.M).group(1)))
p=0; out={}
for load,seed,T in hdr:
    loc=np.frombuffer(raw[p:p+T*8],dtype=np.int8).reshape(T,8); p+=T*8
    c=np.bincount(loc[loc>=0].astype(int),minlength=32); c=c[c>0]
    out[(load,seed)]=sorted(c.tolist(),reverse=True)
    if T in (1,8,16,32) and seed in (7,11,13,17,99):
        g8=int(np.sum((c+7)//8)); g16=int(np.sum((c+15)//16))
        print(load,seed,'E',len(c),'counts',sorted(c.tolist(),reverse=True),'g8',g8,'g16',g16)
