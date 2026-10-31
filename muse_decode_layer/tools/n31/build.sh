#!/bin/sh
# usage: build.sh <out> -DSWAP=.. -DTM=.. ...   (CUTLASS headers from the system tilelang wheel)
o=$1; shift
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_90a --expt-relaxed-constexpr -I/usr/local/lib/python3.14/site-packages/tilelang/3rdparty/cutlass/include -I/usr/local/lib/python3.14/site-packages/tilelang/3rdparty/cutlass/tools/util/include "$@" /home/daytona/r-layer/muse_decode_layer/tools/n31/cl_gemm.cu -o $o 2>&1 | grep -E "error" | head -5
