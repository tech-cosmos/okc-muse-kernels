#!/bin/sh
# x24 (after 00:00Z, ~11 Modal GPU-min): H200 standalone muse_attention_decode A/B, ours (~/attn main) vs vjeux's public
# muse_attention_decode.py (github.com/vjeux/okc-2026, Apache-2.0; Madhavan113's 2.20x layer embeds it). Question: is
# vjeux's flat q8/q1 full-layer kernel faster than our _gw_verify/_gd_full on the H200 (judge per-kernel profiles say
# v32 full 184.6 vs 190.7 us med + our _combine_bal tail)? All 12 shapes print; read the full ones. Run from ~/r-fresh.
set -e
D=/tmp/x24h; rm -rf $D; mkdir -p $D/ours $D/vj
for d in ours vj; do cp ~/attn/*.py ~/attn/tools/bench.py $D/$d/; chmod u+w $D/$d/*; done
test -f /tmp/pub/vjeux_okc-2026/muse_attention_decode.py || git clone -q --depth 1 https://github.com/vjeux/okc-2026 /tmp/pub/vjeux_okc-2026
cp /tmp/pub/vjeux_okc-2026/muse_attention_decode.py $D/vj/submission.py
. ~/okc/bin/activate
python tools/h200.py $D/ours base=$D/ours/submission.py cand=$D/vj/submission.py --rounds 1 --cmd "python bench.py bench"
