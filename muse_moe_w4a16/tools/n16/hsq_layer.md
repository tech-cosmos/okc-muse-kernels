# n16: tk2h (GS16 + PF4) on the SQ single-copy stages -- port into r-layer moe_src.py

The SQ stage image (_sq_pack) is byte-identical to the 3-copy stage, and tk2h shares tk2f_produce, so tk2h on SQ
is the same kernel with PF | 512 and the w13sq/w2sq buffers (bitwise = 3-copy: 348/348, tools/n16/hsq_t.py).

1. moe_tk2f_launch, inside `if (sq) {` before its TORCH_CHECK:
     if (cfg >= 140000 && cfg < 150000 && cfg % 1000 == 4)  // tk2h GS16 + PF4 on SQ stages
       return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4 | 512>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
2. _tk2f_w: also map c % 1000000 in [140000, 150000) with c % 1000 == 4 to (c + 10000000, w13sq, w2sq).
   (The layer's tk2h cfg = 100000 + 40000 + W * 1000 + 4; moe_norm3 after it, unchanged.)
