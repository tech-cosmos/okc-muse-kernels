# s47: list loads that precede the first griddepcontrol.wait (SASS ACQBULK) per kernel of a built .so.
# usage: PATH=/usr/local/cuda-13.0/bin:$PATH python dev/s47_pdl_audit.py <lib.so>  (then map each load's base pointer to a param)
import re,sys,subprocess
so=sys.argv[1]
out=subprocess.run(["cuobjdump","-sass",so],capture_output=True,text=True).stdout
funcs=re.split(r"\n\s+Function : ",out)[1:]
for f in funcs:
    name=f.split("\n",1)[0].strip()
    lines=f.split("\n")
    idx=[i for i,l in enumerate(lines) if "ACQBULK" in l]
    if not idx: continue
    pre=[l.strip() for l in lines[:idx[0]] if re.search(r"\b(LDG|LDGSTS|LD\.|UBLKCP|UTMALDG|LDGDEPBAR)",l)]
    print("==",name[:150],"waits",len(idx),"preloads",len(pre))
    for l in pre: print("   ",re.sub(r"\s+"," ",l)[:150])
