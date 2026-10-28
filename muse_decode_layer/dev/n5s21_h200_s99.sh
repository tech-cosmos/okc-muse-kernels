# n5s21: Modal H200 runner (tools/h200.py --cmd "bash x99/run.sh"; overlay dir = submission.py + x99/ = dev/s99_g30.{py,cu} + this file as run.sh)
set -x
SH="unp:8192:4096:1 unp:8192:4096:2 pool:4096:8192:2 pool:4096:8192:4"
DEFS="" python x99/s99_g30.py $SH 2>&1 | grep -E "err|Error|error" 
DEFS="-DG30_R2=1 -DG30_NS=10 -DG30_NW=5" python x99/s99_g30.py $SH 2>&1 | grep -E "err|Error|error"
DEFS="" python x99/s99_g30.py $SH 2>&1 | grep -E "err|Error|error"
