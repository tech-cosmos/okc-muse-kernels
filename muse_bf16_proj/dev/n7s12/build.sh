#!/bin/bash
# build.sh NAME "-DTM=128 -DTN=32 -DSCHED=0 -DMODE=1 [-DCUTLASS_MIXED_DTYPE_E2M1]"
C=/tmp/cutlass
mkdir -p /tmp/n7s12
/usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -DNDEBUG \
  -I$C/include -I$C/tools/util/include -I$C/examples/common -I. $2 ex55.cu -o /tmp/n7s12/$1 2>&1 | grep -v -i "warning\|^$" | head -20
