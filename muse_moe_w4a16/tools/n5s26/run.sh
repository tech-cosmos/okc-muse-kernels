#!/bin/bash
# usage: run.sh : ncu moe_tk2f duration for MAXR x VAR x (load,seed)
cd /tmp/n5s26; . ~/okc/bin/activate
for ls in "decode 16:7" "decode 16:99" "decode 32:11"; do l=${ls%:*}; s=${ls#*:}
for m in 168 224; do for v in sq dt; do for r in 1 2; do
 MAXR=$m VAR=$v flock /tmp/okc-gpu.lock /usr/local/cuda-13.0/bin/ncu --profile-from-start off -k regex:moe_tk2f --metrics gpu__time_duration.sum,smsp__inst_executed.sum,launch__registers_per_thread python ncu_dt.py "$l" $s 2>&1 | awk -v t="$l s$s MAXR=$m $v" '/gpu__time|inst_exec|registers_per/{printf "%s=%s ", $1, $3} END{print " ", t}'
done; done; done; done
