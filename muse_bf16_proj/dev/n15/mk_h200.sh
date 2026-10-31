#!/bin/bash
# proj n15: regenerate the H200 v1 pr8 overlays (base+flat, now, tk1) from r-layer main and run them on Modal.
# Run AFTER 00:00Z (proj Modal share). Needs pr8_n15.diff (PR8_TRACE/TK1/DP defines; trace on by default -> stripped here).
set -e; . ~/okc/bin/activate; D=$(cd $(dirname $0); pwd); N13=$D/../n13s1; W=/tmp/pj15m; rm -rf $W; mkdir -p $W/L
git -C ~/r-layer archive main muse_decode_layer | tar -x -C $W/L; cd $W/L/muse_decode_layer
git -C ~/r-layer show main:muse_decode_layer/submission.py > $W/main_sub.py
patch -s pr8_src.cu $D/pr8_n15.diff; sed -i '/^#ifndef PR8_NOTRACE$/,+2d' pr8_src.cu; ! grep -q "define PR8_TRACE" pr8_src.cu
python build.py > /dev/null
python - "$W" <<'PY'
import pathlib, sys, json
W = pathlib.Path(sys.argv[1]); s = pathlib.Path('submission.py').read_text()
i = s.index("_GPR8_SRC = r'''") + len("_GPR8_SRC = r'''")
for k, v in {"now": "PR8_NOW", "tk1": "PR8_TK1"}.items():
    (W / f"v_{k}.py").write_text(s[:i] + "\n" + "".join("#define " + x + "\n" for x in v.split()) + s[i:])
for k in ["base", "now", "tk1"]:
    (W / "h200" / k).mkdir(parents=True, exist_ok=True)
    (W / "h200" / k / "varj.json").write_text(json.dumps([["base", {}], ["flat", None], ["base2", {}]] if k == "base" else [["base", {}], ["base2", {}]]))
PY
python $N13/orc8.py $W/main_sub.py $W/h200/base/submission.py
for k in now tk1; do python $N13/orc8.py $W/v_$k.py $W/h200/$k/submission.py; done
for k in base now tk1; do cp $N13/tl13b.py $W/h200/$k/; done
[ "$1" = "--dry" ] && { ls -R $W/h200; exit 0; }
python ~/r-layer/tools/h200.py $W/L/muse_decode_layer base=$W/h200/base now=$W/h200/now tk1=$W/h200/tk1 \
  --cmd 'LOAD="verify 1" TOK=8 KN="pr8::" VARJ="$(cat varj.json)" python tl13b.py submission.py 7 2>&1 | grep RES'
