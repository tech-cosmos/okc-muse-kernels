#!/bin/bash
# usage: ncu_t.sh bs cfg  -> per-kernel durations at ncu's locked base clock (H200 proxy)
cd ${DIR:-~/r-moe/muse_moe_w4a16}
/usr/local/cuda-13.0/bin/ncu --profile-from-start off --metrics gpu__time_duration.sum python ~/r-moe/muse_moe_w4a16/tools/sk_ncu.py "$@" 2>&1 | awk '/^  [a-z_ ]*[a-z_]+[<(]/{name=$1" "$2} /gpu__time_duration.sum/{printf "%s %s | ", substr(name,1,22), $3; tot+=$3} END{printf " TOTAL %.1f\n", tot}'
