#!/bin/bash
# usage: ncu_tk2f.sh <bs> <cfg...>: moe_tk2f duration (ncu base clock = H200 proxy), instructions, issue active
cd ~/r-moe/muse_moe_w4a16; bs=$1; shift
for c in "$@"; do
  TK2F=$c /usr/local/cuda-13.0/bin/ncu --profile-from-start off -k regex:moe_tk2f --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed python tools/ncu_tk2f.py $bs 2>&1 | awk -v c=$c '/gpu__time|inst_exec|issue_active|dram__/{printf "%s=%s ", $1, $3} END{print " cfg", c}'
done
