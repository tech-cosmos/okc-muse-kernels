# n5s21: Modal H200 runner (overlay x101/ = dev/s101_g30{,real}.py, s101_g30.cu, s101_g30split.cu + this file as run.sh)
W=qkv_gate SPLIT=1,3,2 python x101/s101_g30real.py 2>&1 | grep "pdl 1"
W=pooler SPLIT=2,3 python x101/s101_g30real.py 2>&1 | grep "pdl 1"
python x101/s101_g30.py qkv:2304:8192:3 qkv:2304:8192:2 pool:4096:8192:2 pool:4096:8192:3 2>&1 | grep err
W=qkv_gate SPLIT=3 python x101/s101_g30real.py 2>&1 | grep "pdl 1"
