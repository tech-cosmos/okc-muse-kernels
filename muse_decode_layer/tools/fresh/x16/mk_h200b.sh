#!/bin/sh
# x17 (after 00:00Z): Modal H200 A/B v176 (b051588) vs v177 (51eaeff), tuner pinned off in both (TUNE = {} and ...),
# one load group per Modal call (each ~3-4 GPU-min; fresh's share 20/day). Run from muse_decode_layer/ in ~/r-fresh.
# Order of value: decode 32 (MRG 2 + NDS_PW + CB_HB_TRIG), verify 16 (NVS_PW), decode 16 (GD_PW), verify 32.
set -e
D=/tmp/x17h2; mkdir -p $D
git -C ~/r-layer show b051588:muse_decode_layer/submission.py | sed 's/^TUNE = {1: /TUNE = {} and {1: /' > $D/v176.py
git -C ~/r-layer show 51eaeff:muse_decode_layer/submission.py | sed 's/^TUNE = {1: /TUNE = {} and {1: /' > $D/v177.py
grep -c "^TUNE = {} and" $D/v176.py $D/v177.py
. ~/okc/bin/activate
IFS=,; for L in $LOADS; do
  echo "=== $L"
  python ../tools/h200.py . base=$D/v176.py cand=$D/v177.py --rounds 2 --cmd "python bench.py bench submission.py '$L'" 2>&1 | grep -v "^$"
done
