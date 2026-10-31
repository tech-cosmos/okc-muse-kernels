#!/bin/sh
# x35 (after 00:00Z, ~3-4 GPU-min): Modal H200 A/B of v183 (e0dd26f) vs branch fresh f064596 (CSL_PW at T=1 minus the csl
# smem carveout), tuner pinned off in both, 'decode 1 sliding' only. Run from muse_decode_layer/ in ~/r-fresh.
set -e
D=/tmp/x35h; mkdir -p $D
git -C ~/r-fresh show e0dd26f:muse_decode_layer/submission.py | sed 's/^TUNE = {1: /TUNE = {} and {1: /' > $D/v183.py
git -C ~/r-fresh show f064596:muse_decode_layer/submission.py | sed 's/^TUNE = {1: /TUNE = {} and {1: /' > $D/x35.py
grep -c "^TUNE = {} and" $D/v183.py $D/x35.py
. ~/okc/bin/activate
python ../tools/h200.py . base=$D/v183.py cand=$D/x35.py --rounds 2 --cmd "python bench.py bench submission.py 'decode 1 sliding'" 2>&1 | grep -v "^$"
