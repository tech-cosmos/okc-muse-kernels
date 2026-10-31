cd ~/r-moe/muse_moe_w4a16; . ~/okc/bin/activate; export PYTHONDONTWRITEBYTECODE=1; D=~/r-proj/muse_bf16_proj/dev/n21
S=1,4,12,17,22,25,32,38,0,16,20,39
timeout 1200 python $D/ab_moe.py $D/base $D/base2 256 $S 0 4 2>&1 | grep "seed\|bs " | sed 's/^/AA /'
timeout 1200 python $D/ab_moe.py $D/base $D/sl2 256 $S 0 4 2>&1 | grep "seed\|bs " | sed 's/^/SL2 /'
