#!/bin/bash
# proj n24: H200 power/clock test of the judge tuner's T=256 'confirm slower' effect (25/25 judge logs: T=256 full confirm
# +10..20 us vs first pass on the same graphs). Run AFTER 00:00Z (proj Modal share), ~3-5 GPU-min. Layer = r-layer main.
set -e; . ~/okc/bin/activate; D=$(cd $(dirname $0); pwd); W=/tmp/pj24m; rm -rf $W; mkdir -p $W/L $W/ov
git -C ~/r-layer archive main muse_decode_layer | tar -x -C $W/L
cp $D/conf.py $W/ov/
[ "$1" = "--dry" ] && { ls -R $W/ov; exit 0; }
python ~/r-layer/tools/h200.py $W/L/muse_decode_layer base=$W/ov \
  --cmd 'python conf.py . submission.py "verify 32" 2400 2>&1 | grep -v Warn | tail -16'
