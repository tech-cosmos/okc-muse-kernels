#!/bin/bash
# build.sh NAME "-DTM=.. -DTN=.. ..." -> /tmp/fcg/NAME.so
NAME=$1; shift
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_90a --expt-relaxed-constexpr -Xcompiler -fPIC -shared \
  -I/tmp/cutlass/include -I/tmp/cutlass/tools/util/include -DNDEBUG $@ \
  -o /tmp/fcg/$NAME.so $(dirname $0)/cg.cu > /tmp/fcg/$NAME.log 2>&1 && echo "ok $NAME" || echo "FAIL $NAME"
