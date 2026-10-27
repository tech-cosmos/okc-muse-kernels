#!/bin/zsh
# usage: ptx.sh -> builds submission's lib, prints C7512 lines for moe_twq and the twq loop shapes
cd ~/r-moe/muse_moe_w4a16; . ~/okc/bin/activate
f=$(python -c "import submission as m; m._lib(); print(m._LIB.__file__)" 2>/dev/null | tail -1); D=$(dirname $f)
F=$(grep "^cuda_cflags" $D/build.ninja | sed 's/^cuda_cflags = //')
cd /tmp/s25; /usr/local/cuda/bin/nvcc ${=F} -c $D/cuda.cu -o /tmp/s25/k.o -Xptxas -v > ptx.log 2>&1
grep C7512 ptx.log | grep -c moe_twq
/usr/local/cuda-13.0/bin/cuobjdump -sass $f > cur.sass; python3 lp.py cur.sass
