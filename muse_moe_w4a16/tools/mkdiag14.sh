#!/bin/sh
# s14: H200 diag copy of submission.py (never submit it): DIAG variants (twm orders) + twm timelines (TWM_TL build).
# usage: sh tools/mkdiag14.sh /tmp/moe14/d2 && cd /tmp/moe14/d2 && okc check submission.py --json > chk.json
# then: grep -h "diag12 MEAN\|diagtl" profiles/*.log
mkdir -p "$1"
sed 's/^DIAG = 0$/DIAG = 1/; s/^#define TWM_TL 0$/#define TWM_TL 1/; s/^DIAG_TL = () /DIAG_TL = (0, 76) /' submission.py > "$1/submission.py"
grep -c "^DIAG = 1\|^#define TWM_TL 1\|^DIAG_TL = (0, 76)" "$1/submission.py"
