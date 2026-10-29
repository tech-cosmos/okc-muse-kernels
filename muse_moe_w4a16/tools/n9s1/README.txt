n9s1 (2026-10-29 23:15Z-): lead 22:30Z step (1), in-layer twm attribution. Base code = r-fresh b8888bd muse_decode_layer
(TWM_ORC switch by fresh). Recreate the variants (in a copy of that dir):
  for o in 0 1 7 8 256; do sed "s/^#define TWM_ORC 0 /#define TWM_ORC $o /" submission.py > orc$o.py; done
  L2-alias (every expert's weight address -> expert 0, 28.3 MB, L2-resident on the H100 = bandwidth not binding):
  sed '7490,7850s/(size_t)e \* /(size_t)(e * 0) * /' orc0.py > al0.py     (line range = moe_twm/moe_hot/flat-read in b8888bd)
  python inst.py orc0.py i_orc0.py   (+ the globaltimer lines: see NOTES-moe n9s1) ; python run_inst.py i_orc0g.py "verify 16 sl,verify 32 sl"
Counters (lane 0 of each of the 8 consumer warps, summed over CTAs, reported per warp): life (consumer start->loop end),
between-item wait (outer mbar_wait for an item's 1st stage: ring fill, g2 hand-off spin), item (TWI call), loop (tw_item
main loop), ring wait (mbar_wait(full) inside the loop), epi+rest = item - loop. v32: moe_hot also calls tw_item, so loop/ring
wait/items include moe_hot's warps there (epi+rest negative): use life/between/item only at v32.
