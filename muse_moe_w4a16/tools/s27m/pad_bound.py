import sys, zlib, base64, re, collections, math
src=open('submission.py').read()
R=re.search(r'^_DIAG_R = "([^"]+)"',src,re.M).group(1)
hdr=eval(re.search(r'^_DIAG_HDR = (.*)$',src,re.M).group(1))
raw=zlib.decompress(base64.b64decode(R)); p=0
t=lambda N: 43+19*(N-8)/24   # H200 us per 1 MB RT4 item (s24 timeline: N8 43, N16 49, N24 58, N32 62)
ITEMS=27
for load,seed,T in hdr:
    b=raw[p:p+T*8]; p+=T*8
    if T<=32: continue
    c=collections.Counter(x for x in b if x<128)
    sav=0; base=0
    for e,n in c.items():
        if n>32: continue
        Np = 8 if n<=8 else 16 if n<=16 else 32
        Ni = 8*math.ceil(n/8)
        sav += (t(Np)-t(Ni))*ITEMS/132; base += t(Np)*ITEMS/132
    print(load,seed,f"RT4 experts time {base:.1f} us, exact-N8 saving {sav:.2f} us")
