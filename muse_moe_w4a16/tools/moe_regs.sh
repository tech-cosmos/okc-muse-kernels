#!/bin/bash
# usage: moe_regs.sh <submission.py>  -> ptxas register/spill summary of the kernels
python3 - "$1" <<'PY'
import sys,re
s=open(sys.argv[1]).read()
src=s[s.index('CUDA_SRC = r"""')+15:]
src=src[:src.index('#include <torch/extension.h>')]
src += '\nint dummy(int cfg, const void* wq, const void* ws, const void* gs, const void* lut, const void* x, const void* meta, const void* sr, void* out, cudaStream_t st) { if (cfg > 1000) { DISPATCH(true) } DISPATCH(false) }\n'
if 'SK_DISPATCH' in src:
    src += 'int dummy2(int cfg, const void* wq, const void* s2, const void* lut, const void* x, const void* meta, const void* sr, void* acc, cudaStream_t st) { if (cfg > 1000) { SK_DISPATCH(512, true) } SK_DISPATCH(256, false) }\n'
open('/tmp/moe_regs.cu','w').write(src)
PY
/usr/local/cuda-13.0/bin/nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 -c /tmp/moe_regs.cu -o /tmp/moe_regs.o -Xptxas -v 2>&1 | grep -E "Compiling|registers|spill" | paste - - - | sed -E 's/ptxas info *: //g; s/Compiling entry function//; s/for .sm_90a.//' | awk '{print substr($0,1,200)}'
