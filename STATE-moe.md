## o15 16:15Z 10-31: bs1 is 0.64 %/us of the moe board (bs32 0.14, bs256 0.07). Cluster-split norm3 (moe_norm3c, bitwise,
## tools/o15/n3c.diff) H100 bs1 +0.4/+0.7, bs8 +0.5 us = DEAD. 0 judge runs. main still cbd4bcf: SUBMIT IT ONCE AFTER 00:00Z 11-01.
## o14 15:31Z 10-31: main cbd4bcf = acf63a9 + tk2h MRG 2 at bs32 (fresh x13 per-segment counters); bitwise, test 15/15.
## H100 bs32 -2.2 us; H200 DIAG cc253b3b bs32 -0.7 us (16 seeds, 2 pairs). Clean check c015407f 2.3737x = BEST check:
## bs1 25.9, bs16 85.2, bs32 120.0, bs8 61.5, bs128 189.9, bs256 229.4. NEXT SESSION AFTER 00:00Z 11-01: SUBMIT cbd4bcf ONCE
## (supersedes acf63a9/7139706). moe board 10-31: 9 checks, 2 submits. Dead: tk2h at bs16 (+2.5 H100), BAL at bs32 (tie).
## o13 15:30Z 10-31: main acf63a9 = 7139706 + tk2f BAL on SQ (PF 548, layer T=16 'bal') at bs 8/16; bitwise, test 15/15.
## H200 DIAG 51ba444e (moe-board routings, interleaved): bs16 -1.1, bs8 -0.3 us. Clean check bfd582aa 2.3694x (best check):
## bs1 26.4, bs16 84.8, bs32 117.2, bs8 62.1, bs128 193.4, bs256 228.4. NEXT SESSION AFTER 00:00Z 11-01: SUBMIT acf63a9 ONCE
## (not 7139706). moe board 10-31: 7 checks, 2 submits. Layer T=8 (nu8) does not map; HOT_ET still unported (o11 race note).
## H200 moe-board A/B harness: tools/o13/diag_bal.diff (O13_DIAG hook; copy, edit variants, okc check the copy, read log_tail).
## o12 15:40Z 10-31: main 7139706 = c75be69 + moe_hot HOT_XN 16 (layer v172), bitwise, test 15/15. check 2ba95722 2.3265x
## (bs256 x2.728 vs 2.683 twice before = real; others slow draws). NEXT SESSION AFTER 00:00Z 11-01: SUBMIT 7139706 once
## (moe board cap resets; beats 2.3198). Then: HOT_ET port (race-safe form) is the last unported layer moe piece.
## o11 10-31: moe board = layer moe_hot hotdyn+HOT128 (HOT_M {128: 16048, 256: 3216064}) + bs32 tk2h W3 (TK2H_M {32: 3}) on main c75be69,
## bitwise, test 15/15. Checks e59302f4 2.3275 (hotdyn only) / b4e94b61 2.3323; submit 41ad7717 drew 2.3124 -> board stays 2.3198 (e37e578e).
## moe board 10-31: 4 checks (1 DIAG), 2 submits = lead's 2/agent/day cap reached. Do NOT resubmit c75be69 unchanged.
## CW (fresh a269066) on the moe board: H200 DIAG 3e70d87c d1 +0.2 us = DEAD (patch only in /tmp/o11cw; NOTES o11).
## NEXT: moe-board candidates need new code (gap per shape vs 4.3 TB/s bytes: bs32 +15, bs16 +12, bs8 +11, bs1 +8, bs128 +50, bs256 +77 us;
## distinct experts mean bs1 2.7 / bs16 11.4 / bs32 15.7 / bs8 8.0 / bs128 21.8 / bs256 23.9). Layer lane items per lead lines.
## o10 14:20Z 10-31: fresh x10 'late' on moe board: check bef7bd02 2.3201x, SUBMIT e37e578e 2.3198x = NEW MOE BOARD BEST
## (was aeba42c5 2.2792): bs1 25.3, bs16 87.3, bs32 118.6, bs8 63.2, bs128 194.0, bs256 234.3 us. main f90bcaf. moe 10-31: 1 check, 1 submit.
## NEXT (14:25Z 10-31): o10 done. Big-M twm PDL trigger stays at entry (NORM_EARLY 1 = moe_norm waits per expert during twm: early is right). Candidates: fresh CW handoff (bs1 ~1 us) -> moe check; else lead names one. Slack: PARKED-READY (moe/slack a9242fb;
## o8 cluster kernel tools/o8/s3.cu, bitwise, untimed, ~2-3 us of the 6-7 us overhead if approved).
## o9 13:40Z 10-31: lead 12:25Z TWM NON-DQ OVERHEAD: GATE FAILS (tools/o9, real tw_item harness, H100). 269 instr/stage RT4 N32 =
## dq 192 + 16 HGMMA + 8 WG + ~19 uniform + ~34 vector. Exact hoist (empty-barrier address, -4 instr) tie; whole ring protocol free
## (illegal) 0.945x RT4 N32 / 0.970x RT2 N64; + scales free 0.924/0.913; half the full waits tie. 0 judge runs, 0 Modal.
## o8 12:45Z 10-31: slack cluster kernel built (bitwise = o6 on 270 cases), stopped untimed per lead 12:25Z. NOTES o8.
## o7b 12:12Z: layer's request tk2f LAST-WAVE K-SPLIT: DEAD at d16 (oracle: removing the row tail saves less than its bytes).
## o7 12:05Z 10-31: lead 11:25Z tk2f FIXED COST: GATE FAILS. Free-prologue+split oracle (plan of the same input in one trip)
## H100 in-layer: v1 +0.0/+0.5, d16 -0.2/-0.7, d32 -0.7/-0.9 us; the prologue is one ~1.3 us dependent trip (TLB+L2), its compute
## is free; LUT-late dead. d16 lever = _routing's 2.9 us tail after the router (proj's item). Tools tools/o7. 0 judge runs.
## (o6 ran against the lead's 11:25Z PARK of moe/slack, which I missed; slack parked now.)
## o6 11:50Z 10-31: moe/slack now a NET WIN on the H100 (still OFF, 0 judge runs): o5's loss was the kernel's serialized
## global loads (phase C 10 us per changed token), not PDL. Loads-first token_keys + score by shuffle (bitwise = o5) + PDL:
## in-layer on-off v16 sl/fu -3.9/-10.5, v32 -2.1/-2.2 us; 0 fails/288 slack_judge; 13/13. Branch moe/slack a9242fb (on v174).
## Remaining overhead +5.8..+7.0 us (expert saving -9..-17): next = cut it (fuse phase A into _routing / fewer trips).
## o5 11:30Z 10-31: lead 10:25Z ROUTING-SLACK BUILD PREPARED: branch moe/slack 894de97 (slack_src.cu + _slack at all routing sites,
## OFF by default). Greedy = fresh's exact min on 12/12 loads with reference scores, BUT the layer's keys carry FP8-query noise
## (keep-lead std 0.0048, max 0.019 = E4M3-ref vs FP32) -> safe margins leave ~1 expert v16/v32, ~0 d1/d16/v1. H100: 0 fails/192
## inputs; A/B vs main v16 +6.4/+1.6, v32 +4.8/+8.9 us = NET LOSS (kernel 4.4/8.7 us + lost PDL). x1.096 needs reference-exact
## scores; usable slack for us ~0.1-0.3 % even fused. Next if wanted: fuse phase A into _routing + B/C in its last program. 0 judge runs.
## o4 10:30Z 10-31: no moe item >= gate left (o1-o3 closed lead 09:55Z items + own list). New angle checked on paper: x1's acc-read
## wall is per weight-token (BF16 wgmma k16 fixed: one FP32 acc read per token per 16 K for any M/N/RS/SS shape; k32 = FP8 x = illegal),
## so no tile shape escapes it. Sub-gate leftovers unchanged (tk2f last-wave K-split ~0.8/1.1 us d16/d32; twm instr trims <1; HOT128 0.9 v16).
## Largest open lever = routing-slack ruling (fresh x8, x1.096) -> human/organizers. 0 judge runs, 0 Modal.
## o3 10:50Z 10-31: own item SCALE-D-0 PROMOTION in twm (wgmma scale-d 0 per k16 + FP32 FFMA block scale, from fresh x1): DEAD
## by H100 oracle (tools/o3): bound w/o promotion 0.81-0.86x, promotion ALU (N/2 FFMA/tile/k16) 1.25-1.8x even ideal; ptxas C7514
## trap documented (NOTES o3). Nothing in the moe lane >= gate. 0 judge runs, 0 Modal.
## o2 10:35Z 10-31: own-list oracles DEAD: power/clock lever (n9s1: dq+wgmma-free twm clocks 1.80 GHz = base -> clock not compute-set),
## GPC static imbalance in tk2f (inside n7s41 tail bound <= 2.2 us d32), twm structure (= drain, dead on record). NOTES o2. 0 judge runs, 0 Modal.
## o1 10:30Z 10-31: lead 09:55Z PIPE-BALANCED DQ: GATE FAILS by oracle (tools/o1: twm w/o HMUL2 0.87-0.93x = bound; HMUL2->1 IADD
## 0.99-1.03x; exact int product >= 3-4 ops/pair -> slower). Next item tk2f d16/d32 rows: H200 record n7s41 says <= ~1 us/item -> not built.
## 0 judge runs, 0 Modal. NOTES o1.
## n11c 04:36Z 10-30 idle: no lead line after 03:50Z; LANE A done (n11a: 'a legal faster design exists'); moe paused. 0 judge runs, 0 Modal.
## n11a 04:20Z 10-30: lead 03:50Z LANE A DONE (diag only, NOTES n11a, tools/n11a): leaders' edge is NOT an FP8 shortcut:
## n11b 04:25Z 10-30 idle: LANE A done (n11a); no lead line after 03:50Z; moe paused. 0 judge runs, 0 Modal.
## FP8 x alone fails 1.5% (2.6% rms/GEMM; W fold + x 3.4%); cheapest scaled FP8 consumer (F2 LUT fold) only 0.88-0.92x twm
## cycles (bs128/256 -> ~188/~225 us proj. vs #1 ~155/188): dq ALU is the wall in both. Only scale-free F0 (0.55x) reaches it.
## => 'a legal faster design exists'. moe paused. 0 judge runs, Modal 1.1 GPU-min.
## n10s4 04:05Z 10-30: lead 03:20Z bundle DONE: v173 (= v172 + IDW + CUDA bnd2 at T=8) on main 8841e7d; check 335f3f26 2.1657 raw /
## 2.1398 fixed-SGLang (< v172 2.1607): NOT submitted. H100 A/B d16/d32 -0.5, v1 -0.4 us. moe PAUSED. 1 layer check, 0 Modal.
## n10s3 03:12Z 10-30 idle: still no lead line after 02:45Z (END lane closed n10s1; IDW Handoff moe/end 78898bb open). moe paused. 0 judge runs, 0 Modal.
## n10s2 02:57Z 10-30 idle: no lead line after 02:45Z; LAYER END lane closed in n10s1 (its '04:15Z' stamp = typo, ended ~02:53Z).
## moe paused. Open: IDW Handoff (moe/end 78898bb) for the lead/layer to bundle. 0 judge runs, 0 Modal.
## n10s1 04:15Z 10-30: lead 02:45Z LAYER END lane DONE (NOTES n10s1). Free-piece oracles (H100): second boundary d1 1.2-1.4,
## d16 1.8-2.7, d32 0.7-2.2, v1 2.8-3.3 us (cost = post-wait latency chain, not the hop); norm2 d16 1.1-2.3, d32 1.2-1.4 (not y32
## volume: Q4 oracle 0). Buildable: IDW (branch moe/end 78898bb, exact, ~-0.5 us d16/d32: bundle item), CUDA bnd2 at v1 -0.55
## (tools/n10s1, not on a branch). Gate (>= 2 us d1 or >= 3 us summed) NOT met by any design -> moe pauses. 0 judge runs, 0 Modal.
## n9s4 00:30Z 10-30 idle: still no lead line after 00:00Z; moe paused (n9s2). 0 judge runs, 0 Modal.
## n9s3 00:15Z 10-30 idle: no lead line after 00:00Z (answered by n9s2; n9s2's '00:40Z' stamp = typo, it ended ~00:10Z). moe paused.
## 0 judge runs, 0 Modal. Budget 10-30 untouched (moe 20 checks / 10 submits).
## n9s2 00:40Z 10-30: fresh's H200 O2 READ: base-both-free v16 sl 28.9 / v32 sl 47.5 us -> lead gate PASSES on numbers, but dq (14.4/22.4)
## and MMA (11.7/22.7) are fully serial in-layer on the H200 too, and no Hopper operand path escapes the RF/issue wall (RS: dq in the
## issuing warp; SS: ~4.5 B/weight smem). Twm consumer rebuild CLOSED by mechanism (NOTES n9s2). No build, 0 judge runs, 0 Modal.
## NEXT: lead decision. moe pauses unless the lead names a mechanism outside RS/SS.
## n9s1 00:06Z 10-30: lead 22:30Z twm REBUILD step (1) DONE. H200 counters (Modal): twm consumer-bound (real HBM = L2-fed
## within 3%, ring wait 5% of consumer life, SM 1.77 GHz); H100: structure ~2 us, NSTG 6 = 8, g2 spin <= 1.7 us. Gate (lead 00:00Z)
## = fresh's H200 O2 (base - both-free) >= 25 us v16 / 35 us v32; my estimate 12-17 us v16 -> likely CLOSED -> moe pauses.
## NEXT: read fresh's O2 (r-fresh /tmp/fresh12/h200.txt or COORD); if gate passes, design a consumer rebuild vs n7s47/48 dead list.
## Tools: muse_moe_w4a16/tools/n9s1 (inst.py counters, run_inst.py, L2-alias recipe in README). 0 judge runs; Modal 3.5 min.
## n7s55 17:20Z idle: moe PAUSED per lead 16:45Z (bundle closed n7s53, no v172). No newer lead decision. 0 judge runs, 0 Modal.
## n7s54 17:05Z idle: moe PAUSED per lead 16:45Z (bundle done n7s53: no v172). No newer lead decision. 0 judge runs, 0 Modal.
## n7s53 17:05Z: lead 16:45Z bundle -> RT2 D4 DEAD in layer (ptxas C7511 in moe_twm, H100 +26..35 us; moe_hot-only tie), HOT128 no-hot
## +2.5 v16 full unfixable cheaply (rg fused), prologue ~0; tk2f items 1/2 unbuilt (<=1 us each). No v172, 0 runs. moe paused.
## n7s52 16:25Z idle: no lead decision after 15:50Z (Humming done n7s50). 0 judge runs, 0 Modal. Waiting for a lane or ~/STOP-moe.
## n7s51 16:10Z idle: no lead decision after 15:50Z (n7s50 done; its "16:40Z" stamp = typo, ended ~16:05Z). 0 judge runs.
## n7s50 16:40Z: lead 15:50Z Humming yardstick DONE (tools/n7s50/RESULTS.txt): Humming loses everywhere; H200 its 2 GEMMs alone
## = 1.4-1.7x our full call (bs1 47.5 vs 31.1, bs8 87.7/63.0, bs16 122/80, bs32 183/108); H100 1.03-1.66x. twm/tk2f unchanged. 0 judge runs.
## n7s48 15:35Z: lead 14:45Z (f)(g)(h) done (tools/n7s48/RESULTS.txt): 4 WGs/SM gain 6-13% only at 128 regs w/o producer;
## n7s49 15:45Z idle: no lead decision after 14:45Z; moe lane closed (overlap n7s47/48). 0 runs. Pause moe or name a lane.
## producer warp -> spills; self-fed rings cost ~300 cyc/stage per bulk issue => DEAD. (g) = n7s47 D4. (h) no C75xx in layer moe lib. twm unchanged. 0 runs.
## n7s47 15:35Z: lead 14:30Z overlap (a)-(e) measured (tools/n7s47/ovl2.py, RESULTS.txt; NOTES n7s47): twm = 0.83x of ALU+MMA
## already (bar met), but 1.4x of max(ALU,MMA); main stall = DISPATCH on dq ops while HGMMA runs (ncu). No variant closes it;
## RT2 A-ring D4 -2..-9% per RT2 stage (<= ~1 us/v16 call): bundle item, not built. 0 judge runs.
## Session n7s44 (12:52Z): v171 layer check 46362a70 2.1556x (fixed-SGLang 2.1348), not submitted; moe PAUSED (lead 12:45Z). Bundle list in COORD Requests.
## n7s35 10:33Z idle (no lead decision since 09:35Z; Handoff moe/layer-hotdyn 130a692 pending layer). 0 runs.
## n7s40 11:45Z idle (no lead decision after 11:30Z). Layer board #55 2.158x. HOT_ET branch still a bundle item. 0 runs.
# STATE-moe
## n7s43 12:26Z idle: still no lead decision after 12:00Z. 0 runs.
## n7s42 12:12Z idle: no lead decision after 12:00Z (answered in n7s41). Remaining moe items each <= ~1 us/call. 0 runs.
## n7s41 12:15Z: lead 12:00Z (1) DONE: HOT_ET merged into r-layer main d9553b2 = v171 (build reproduces, test 13/13; no check alone).
## (2) d16/d32 stream lane CLOSED on existing H200 data (DIAG 81391142 re-fit, tools/n7s41): marginal 6.54 us/expert = 4.33 TB/s
## (~96% of stream asymptote); lead's 7.8 us was an H100 layer slope. Left: ph2 extra rows (~1 us/extra group), quant tail
## (<= 1.5/2.2 us d16/d32 upper bound, ~half reachable). Nothing >= 4 us/call. 0 runs. Waiting on the lead for a lane.
## n7s39 ~12:25Z: self-picked HOT_ET (moe_hot triggers twm before its wait; twm polls a flag): r-layer branch moe/layer-hotet 4a9f62d,
## bitwise, test 13/13; H100 pinned A/B v16 sl +1 / full -4, v32 -1.4/-1 us (~-0.05%): BUNDLE item, not on main. 0 runs. NOTES n7s39.
## n7s38 11:55Z: lead 11:30Z v1/d16 regression = NONE (H100 ab_files v168 vs v170 tie on v1/d16; judge picks identical; v169/v170 ratios inside
## the 19-run range, SGLang drew slow too). v170 stays on r-layer main. 0 runs. Nothing queued.
## n7s37 11:10Z idle: no lead decision after 10:40Z (all done). d16 variance closed on paper (= d32 pure bytes, <= 16 tok/expert). 0 runs.
## n7s36 11:06Z: lead 10:40Z (1)-(4) DONE. r-layer main = v170: v169 (hotdyn, T=256 moe_hot device-side CTAs) + v170 (moe_hot at
## T=128, HOT128 32064 = 32 CTAs per 65..128-token expert, <= 64, 0 without). Both bitwise, validate ALL OK. Checks 8b3f8e57 2.1287
## (v169) / 8cea7492 2.1250 (v170, router w/o T=128 hot expert): < board 2.158, not submitted. v16 = bytes + 12 us per hot expert
## (fixed by v170); d32 = pure bytes. Budget 10-29: layer checks 8/20, submits 3/10; moe checks 9/20. Nothing queued.

## n7s34 10:20Z idle (no new lane). NOW (n7s33 ~10:15Z): task 2 (tk2f d1/v1 prologue) DROPPED without a check: in-layer, TK2FX already puts routing + W ring
## fill before tk2f's only griddepcontrol.wait; layer n7s8 IDF oracle (prologue fully hidden) = tk2f end - pr1 end 34.8 vs 35.0
## us (bound ~0); standalone s15 FP = -0.2/-0.4 us H100 (< 0.5 line). Open: Handoff moe/layer-hotdyn 130a692 for layer to integrate.
## Nothing else queued in the moe lane; budget 10-29 moe checks 9/20.
## WAS (n7s32 ~10:05Z): lead 09:35Z task 1 (layer variance) DONE -> Handoff branch moe/layer-hotdyn 130a692 (T=256 moe_hot
## max(32, 16 x hot experts) CTAs, device-side; bitwise; H100 v32 nh3/nh4 -40/-95 us sl, -22/-83 full; else tie). NEXT = task 2:
## tk2f d1/v1 prologue (everything not depending on ids above griddepcontrol.wait; one warp reads ids + issues the first TMA),
## H200 DIAG A/B, drop if < 0.5 us. Variance verdict + numbers: NOTES n7s32. Worktree /tmp/n7s32/wt = branch moe/layer-hotdyn.
## n7s23 2026-10-29 ~09:25Z: lead 08:50Z tk2f d1/v1 lane CLOSED on the H200 (2 DIAG checks); main unchanged (0006dff kernels)
- XPF (L2 prefetch of the CTA's later gemm2 rows at its phase-1 end) DIAG 798fd574: +0.7 d1/v1, +1.0 d16/d32 us.
- WAVE (two-wave row order, gemm2 of wave 1 before wave 2's barrier) DIAG 87ee0bc0: +1.6 d1, +2.7 v1, +3.1 d16, +5.9 d32.
- => the transition is not HBM idle; tk2f d1/v1 = DRAM latency prologue + bytes at ~4.1-4.2 TB/s. Diff: tools/n7s23/wave_xpf_dead.diff.
- Nothing queued in the moe lane. Asked the lead for a lane (COORD Results n7s23). Budget 10-29: moe checks 9/20, 0 Modal.

## WAS-n7s22 (2026-10-29): lead 07:40Z d1/v1 tk2f fixed-cost breakdown = DONE on the H200 (DIAG 81391142)
- d1 E2: kernel span 18.7 vs 13.9 us ideal stream: prologue to first weight request 2.3 (largest) | first-data latency 1.5 |
  transition 1.5 (ring refills, publish 0.5) | tail 0.3; standalone norm2 gap 1.0 + 2.6-3.0. Full table NOTES n7s22.
- TK2F_IDH 1 (ids load before the slut copy) on main 0006dff: bitwise, test 15/15, H100 + H200 = tie. Not for the layer.
- Next candidate: lean T<=4 routing (one warp, one barrier): bound 0.5-0.8 us d1 only. Tools: tools/n7s22/mktl.py, mkdiag.py.
- Budget 10-29: moe checks 7/20, Modal moe share used up.

## WAS-n7s21 ( 2026-10-29 08:15Z): lead 07:05Z build order (T=256 unpool zero-local-token skip) -> gate built, H200 says DEAD
- Custom variable-N swap-AB wgmma GEMM (tools/n7s21/ug.py, (1,2) cluster W multicast, stmatrix.trans epilogue, exact) on the
  H200 (moe DIAG check 477afcc8): cuBLAS 28.03 | ug NT128 29.22 | NT112 27.81 | NT104 27.33 | NT96 26.78 us. v32 kept 201-212 ->
  NT 104/112 = -0.2..-0.7 us per v32 layer before compaction costs -> ~0. The T=256 unpool is L2->SM bound, not tensor-bound.
  Nothing ported to layer.py/moe_src.py. ptxas C7514 / release.cluster / 2x2-cluster pitfalls: NOTES n7s21 (reusable).
- Budget 10-29: moe checks 6/20, layer 2 checks + 2 submits, Modal 46.4/60 (moe's 30 used up).

## WAS-n7s13 (2026-10-29 ~05:00Z): lead 04:25Z items done; waiting on a new lane (COORD Requests n7s13)
- Layer submit 725fc518 of moe/layer-twm 8309328 = 2.1350x (board stays 2.158, #50). Merged into main 9d2b3a8: layer main =
  v163 + twm EPI/KSC2/REL + TK2F_REL 2 (bench test 13/13). Don't check/submit that code again unchanged.
- R1 gate on LAYER v16/v32 routings (H200 Modal, tools/n7s13/tlb_layer.py): v16 idle 12.7-19.6 %, v32 ~8-20 % (moe_hot SMs
  counted busy) -> FAIL, R1 not built. Remaining items (twm N16 epilogue, moe_hot REL) ~1-2 us/call: below the 4 us bundle bar.
- Budget 10-29: layer 2 checks + 2 submits used, moe 5 checks, Modal 13.6/60.

## WAS-n7s1 (n7s1 end, 2026-10-29 ~02:00Z): main = b39e4d2 kernels (unchanged); layer branch moe/layer-twm 8309328 (lead's call)
- n7s1: REDESIGN R1 gate run on the H200 (Modal, board bs128/bs256 inputs, tools/n7s1/tlb.py): twm SM-idle 8-16 %
  (drain 10-26 us, ramp ~4, gaps ~0.6) < the 20 % gate -> R1 NOT built (net <= ~5-8 us after split tax). R2 ceiling
  <= ~1 us by n5s28's TK2FN result -> not built. Every moe-lane item >= 5 us/call is measured dead (NOTES n7s1 list).
  Lane request posted (COORD Requests n7s1). Budget 10-29: moe checks 5/20, layer 1, submits 0, Modal 11.4/60.

## WAS-n6s6 (n6s6 end, 2026-10-29 ~01:40Z): main b39e4d2 (= 7d2a079 kernels; TK2F_REL 0); layer branch moe/layer-twm 8309328
- n6s6 TK2F_REL (tk2f/tk2h gemm13 publishes by a helper warp; tools/n6s6/patch_rel.py ports it to any copy): judge DIAG
  8c5f1376: tk2h (layer T=32) d32 -0.75 us every seed; tk2f 1-group paths d1/v1 +0.25..0.3, d16 tie. moe default 0; layer
  branch has TK2F_REL 2 (tk2h only), validate ALL OK. ~0.1% layer score: not worth a check on its own.
- Layer branch moe/layer-twm = v163 + twm EPI/KSC2/REL (check 74dcb2bb 2.1569x vs board 2.1580, v16 -6.6/-6.8 us) + TK2F_REL 2.
  Waiting on the lead (COORD Results n6s5): submit under the 09:28Z rule / merge into layer main? We are #49 overall, #50/#51
  at 2.158/2.157: one noise draw away from dropping out of the top 50.
- tk2f release lever: CLOSED (n6s6). Remaining moe items each <= ~2 us/call: (1) twm N16 gemm13 epilogue 2.2-3.8 us/item
  (N24 duplicated gated body, store drain); (2) moe_hot REL. Tools new: tools/n6s6/ab_dec.py (paired H100 decode A/B,
  bitwise), mkdiag_trel.py (decode-only DIAG, 2 libs: judge prepare 225 s), sass_tk2f.py, rundiag.py.
- GIT: worktree /tmp/n6s5/wt is on moe/layer-twm (upstream set to origin/moe/layer-twm in n6s6; before that `git pull
  --rebase` there rebased onto main: never pull in it without checking upstream).

## WAS-n6s4 (n6s4 end, 2026-10-29 ~01:00Z): main 7d2a079 = TWM_KSC_DUAL 1 + TWM_REL 1 (bitwise, 15/15), H200-verified small wins
- Judge H200 DIAGs on layer routings: dual (twm without K-split code for calls with no split) v16 -1.9 / v32 tie; REL
  (helper warp 9 releases gemm13+gemm2 items) v16 -0.5 / v32 -1.4 us per call, every seed. Modal board: bs128 -1.7, bs256 -2.5.
  Plain check a4054bf3 2.2790x = tie with board 2.2792x (decode drew slow) -> not submitted. Handoff posted for the layer.
- H200 twm timeline (a18cfaed): epilogue 6-11 us per CTA per call (was 9-14), gemm13 N8 item end 1.4-2.2 us, N16 2.2-3.8.
- LAYOUT RULE: after ANY twm/tk2f code change run `python tools/n5s17/loop64k.py <k.o> 'moe_twm|moe_hot'` (sass.py builds
  k.o under /tmp/n6s4/sass_<hash>): an MMA loop across 64 KB = +4..12% (n5s17). NM (one tw_item per RT) DEAD.
- DIAG copies: tools/n6s4/mkdiag_ksc.py / mkdiag_rel.py pattern (plain library, big-M routings only: prepare ~216 s).
  NEVER run 2+ DIAG checks at once (34547d56 hit 323 s > 300 s cap while 3 compiled concurrently).
- NEXT (ranked): (1) tk2f release analog (bar.sync 2 + thread-0 fence at each segment end stalls the next row's bar.sync 1;
  no idle warp in tk2f: defer the non-final publishes / merge into the CTA-end publish when !BAL), H200 DIAG on d1/d16/d32/v1;
  (2) remaining twm epilogue (N16 gemm13 2.2-3.8 us/item: N24's duplicated gated body, store drain); (3) moe_hot same REL.
- Board stays 2.2792x (aeba42c5). Checks used 10-29 (after n6s6): moe 5 of 20 (4 DIAG + 1 plain), layer 1; submits 0. Modal 9/60 min.
- Tools: tools/n6s3/ab2.py A.py B.py (paired H100 A/B, board bs128/256 + layer v16/v32, max|A-B|); tools/n6s4/sass.py <file>;
  /tmp/n6s4/rundiag.py <diag copy> (local run of a DIAG prepare). The judge REJECTS `import reference`.

## LANE since 2026-10-27 09:45Z (lead LANE DECISION 2): muse_attention_decode as a problem (dir muse_attention_decode/)
Own budget: 20 checks + 10 scored/day on muse_attention_decode. Targets: batch-1 sliding (csl d1, vtk v1) and verify sliding
first; layer owns the full kernels (_gd_full d16/d32, _gw_verify v16/v32): don't work on those. Winners -> COORD Handoffs
(layer ports into r-layer attn_src.py). Precision: FP8 KV, Q >= E4M3, FP32 partials. ~/attn STATE.md dead-ends index = 126
sessions of history: read it before any idea. Board: 2.5962x (1450.7 us, submit 957cabd7 s33; was 2.566x); #1 pjmullin3 3.032x (1242.3 us).
Per-us value (overall attention score): d1 sl 1.94 %/us, v1 sl 1.48, d1 full 0.67, v1 full 0.51, d16 sl 0.44, v16 sl 0.35,
d32 sl 0.24, v32 sl 0.2. (Score = geomean over 6 loads; load = 46 x sliding + 16 x full.)
H200 per shape (us), 06456b7e (= seed 410dcb6 code, 2.570x): d1 4.34/12.25, d16 7.68/87.03, d32 12.29/167.67,
v1 5.49/16.60, v16 11.45/104.89, v32 15.51/194.99 (sliding/full). Judge records: median_ns = raw - ~4.28 us replay cost.
Tools (run from muse_attention_decode/): tools/bench.py test|bench; tools/jsum.py A.json B.json (per-shape deltas of okc
--json outputs); tools/s31/chk_sc.py (scored sliding shapes vs reference, 3 seeds); tools/s31/csl_ab.py A.py B.py (paired
H100 A/B of the csl/vtk kernels from two files); tools/s31/cslstamp.py [B] (per-CTA clock64 timeline of csl, VAR=... variants).

## Board
- Best submit: **aeba42c5 @a60019d (TWM_EPI), 2.2792x** (21:30Z 10-28; check fc7e86d6 2.3127x). Was 4fb734c6 @eb8c03c 2.2703x. Was 951c17e0 @21deb8a (bs256 hot32+split66), 2.2535x (2026-10-28 05:45Z; check cefcc133 2.2632x). Was febfcb1f @72d1274 (SCB) 2.2469x. Was 6a92803a 2.190x. Board is a TEST BED only (lead 01:45Z): the layer counts.
  Checks 20/20 used on 10-26 (DIAGs since s12: never submit those); budget resets 00:00Z. Per-shape best-ish: bs1 27-29, bs8 67-71, bs16 93-99, bs32 128-135,
  bs128 196-207, bs256 243-251 (judge noise +-4%).
- **H200 layer profiles**: ~/r-layer/muse_decode_layer/profiles/submission.<job>.txt = judge per-kernel times per shape.
  USE THEM (they show the experts inside the layer). moe_norm's time there = mostly twf's tail (PDL secondary).
- **H100 is NOT a proxy** for big-M changes (s11: RT 2 H100 -3..9 us, judge +8%) nor for decode (DRAM-bound locally).
  Decode: ncu x 0.72 / instruction counts; tools/l2warm.py = consumer-bound proxy (1 expert, warm L2).
- The judge clones the prepared state: never keep scratch buffers or raw addresses in the state.

## H200 DIAG tool (s12) -- use it for every decode/big-M A/B
`DIAG = 1` (only in a /tmp copy: `sed 's/^DIAG = 0$/DIAG = 1/' submission.py > /tmp/x/submission.py; okc check /tmp/x/submission.py`)
makes prepare() time DIAG_VARIANTS (python-global overrides) on the layer's embedded routings (5 seeds x d1/v1/d16/d32
+ v16/v32), cold L2, graph replay; output: grep "diag12 MEAN" <cwd>/profiles/submission.<job>.log. Repeatable +-0.5 us.
tools/mktl_diag.py = same for a per-CTA globaltimer timeline (diagtl lines). H200 fit (tk2f + norm2, layer routings):
t = 14.1 + 6.97 * experts + 4.4 * extra 8-token groups (us); 6.97 us/expert = 4.06 TB/s (85% of HBM peak).

## What's on main (submission.py)
s27 (40de385): SCB: scp<P>() decodes 2 e4m3 scales per PRMT+IMAD, blo/bhi broadcasts fold into HMUL2 .H0_H0/.H1_H1 (free) at
every SA scale site. Bitwise = before. H200 diag 5ad98078: v16/v32 twm -5, d16/d32 -1 us. Layer port: tools/s27m/patch_layer_scb.py.
s21: TK2F_L2H = 4 / TWM_L2H = 1: L2 cache hints on tk2f/twm weight copies (compile-time #defines substituted in _lib()).
(Older options/kernels still in the file -- GS16, DYN, TK2F 0, moe_tw/moe_big/moe_tb/moe_sk/moe_tk: see NOTES s3-s10.)
s11: M > 32 now runs **moe_twm** (TWF_MIX = 1): moe_twf with the item shape chosen per expert (RT 2 / 64-token passes,
packing w*b2, for experts with > 32 tokens; RT 4 / 32-token passes otherwise; stage stride = RT 4 stage). prepare packs b2 too.
Options (off): TWF_SPLIT / TWF_SPLIT_M (K-split tail items, FP32 partial + ticket/flag in st["tk"], split starts at an
expert boundary), TWF_RT / TWF_RT_M (whole-call RT 2), tk2f NC 16 (cfg +20000). moe_route orders an expert's tokens
nondeterministically: any per-pass treatment difference makes outputs nondeterministic (see NOTES s11).
s10 (c33acae): RACE FIX in tk2f_consume: a consumer arrives on empty[stage] only AFTER the stage's dq/mma (was: right
after issuing the LDS -> the producer could overwrite the stage before the LDS read it; = the s7 race). Never move the
arrive back up. moe_tk / moe_tk2 / tk2d (DYN) still have the early arrive (not default paths): fix before using them.
s9: TK2SA = TWSA = 1: e4m3 block scales decoded arithmetically (sca<K>(w) = __byte_perm(w,0,0x4K4K)*16 + 0x77007700)
in tk2f_consume and tw_item (moe_twf) when prepare() found every scale positive normal (st["sa"]); else the smem LUT.
moe_twf's choice is a host flag (moe_set_sa, set in kernel_fn). TK2FN = 0: fused norm (moe_tk2fn_launch) kept, dead.
M <= 32 (decode, bs8): 2 launches, moe_tk2f (gemm13 rows then gemm2 rows in ONE persistent launch: one routing
prologue, the ring continues, so the producer streams w2 while gemm13 drains) + moe_norm2.
  Launch cfg = (D+1)*1000 + KPS*100 + NSTG (tk2f_go dispatch in moe_tk2f_launch). Defaults: TK2F = TK2F_DYN = 204 (static, all M:
  the submitted code); DYN = cfg 5402 (judge: no gain, check b890d69b). KPS 2/4 cut per-stage overhead: H200 -3..4% at bs8-32 (check 74a049c0);
  locally the H100 is DRAM-bound (ncu dram 83%) and shows nothing: use smsp__inst_executed (tools/ncu_tk2f.sh) as proxy.
gemm13 consumers add their finished rows to cnt[0] (bar.sync 2 + thread-0 threadfence + atomicAdd); gemm2 consumers
spin (ld.acquire) until cnt[0] == n13, then load inter with __ldg. cnt = st["cnt"] (int32[128], zeroed in prepare,
re-zeroed by moe_norm2 after each call: no memset node). DO NOT go back to per-segment counters or ld.global.cg
inter loads without a many-routing sweep (tools/sweep_routings.py, also with +64 KB dyn smem): see NOTES s7 race.
M > 32 (TW_CFG = 418, TWF = 1): route -> moe_gather -> moe_twf (gemm13 AND gemm2, one persistent launch) -> moe_norm.
- moe_twf<RT=4, NSTG=8> (s6): 384 threads = 2 consumer WGs + producer warp (setmaxnreg 40/232), 1 CTA/SM. Work
  counter meta[66] hands out all gemm13 items, then all gemm2 items (heaviest experts first). Item = (expert,
  32-token pass (TWC::PS), row block of 8 warps x 4 16-row tiles; gated: 2 gate + 2 matching up tiles per warp).
  gemm13 item end: fence.proxy.async + bar.sync 1,256 + threadfence + atomicAdd(done[e] = meta[68+e]); gemm2 producer
  polls done[e] (ld.acquire) == npass*16 before its first TMA of inter. Deadlock-free (only running CTAs take
  items; gemm13 items never wait). meta is 100 ints; moe_route zeroes counters + flags.
- ptxas rule (C7512 "insufficient register resources"): it sizes the wgmma pipeline for the WHOLE function, so the
  largest variant decides: RT=4 needs N <= 32 (hence 32-token passes), and gated + non-gated loops in one function
  serialize everything -> one shared main loop (tw_item<..., DYN=true>), epilogue chosen at run time (tw_epi).
  Always check the SASS: `DEPBAR.LE gsb0, 0x1` in loops, no 0x0 except at item ends.
- Stage (1 k-chunk of 64) = TMA tensor-map copy of the x tile (box 64 k x {8,16,32} rows by nv, SW128) + bulk copy
  of the item's weights (prepare: _tb_pack(rt=4) -> w13b4/w2b4 [e][rb][kc][warp][RT][lane]) + 1 of scales.
  Consumers: dq() into registers = wgmma RS A fragment, wgmma m64nNk16 (N = 8*ceil(nv/8)), ring of 2, wait_group 1.
- moe_gather: xg[pos] = x[sroutes[pos]>>3] with each 64-chunk permuted to the _pack_weight K order. xg / inter
  are per-call torch.empty; tensor maps are encoded inside moe_tw_launch from the passed buffer.
- dequant dq(): o0 = q&M, o1/o2 = rotl(q,10/13)&M, o3 = ((q&0x001B001B)*16648)&M, M=0x81C081C0; 4 x
  bf16x2 mul with LUT scale. 12 instr per 8 weights. (_POS in prepare must match.)

## Local harness
Tools in muse_moe_w4a16/tools/ (copies of the /tmp scripts): ncu_t.sh (H200 proxy), seeds.py (200 routings), cmp.py, moe_regs.sh,
moe_sum.py (judge JSON summary), s4: tw_t.py <TW_CFG...> (big-M correctness + local time bs128/256), seeds_big.py
(125 big-M routings incl. skewed), clone_t.py (cloned-state test), sk_tw.py (ncu driver, TW=<cfg> env).
Judge JSON summary: `python /tmp/moe_sum.py <okc --json output>` (recreate if lost: prints per-shape median vs baseline).
Per-kernel timing: /tmp/moe_k.py <bs> '[cfg,...]' (gemm13/gemm2 alone, cold L2); /tmp/moe_p.py <bs> <cfg> for ncu.
`muse_moe_w4a16/bench.py test|bench|sweep`. Local H100 big-M of main (moe_tw): bs128 244, bs256 304 us.
Routing at seed 7: bs1 3 experts x1 token; bs16 13 experts (max 4 tok); bs32 18 (max 7);
bs8 10 (max 2); bs128 25 (max 28); bs256 26 (max 49).

## Attention: what's on main (r-moe muse_attention_decode/submission.py @ s31 code) vs checked
- = 410dcb6 + agent2 halfpfkr (v32 sl cl, csl/nds pt prefetch, nds K-early) + verify 16 on the pre-cl nvs build (_NVS0) +
  csl/vtk owner fast divides. SUBMITTED s32: 90b7ce85 2.5653x (tie with board 2.566x). The same code timed 2.619 / 2.639 /
  2.589x inside s32's DIAG checks (node noise +-3% whole-run): board best stays 2.566x (06456b7e).
- s36: csl SASS audit: independent work already hoisted under the gathers; csl closed. No open >= 0.1% attention item.
- s40 H200 DIAG d5bf959a (tools/s40/mkdiag.py): random-row gather ceiling ~4.1 TB/s (d32 floor 159, d16 79; dense 144/72).
  Decode full at 92-95% of it (headroom d32<=9, d16<=7 us). Verify full consumer-bound (all-L2 v32 170 / v16 93 vs main 188/102,
  floors 157/80): headroom only via a faster _gw_verify consumer. Full kernels at perfect overlap = ~-4% attention score max.
- s42/s43 (NOTES): T=128/256 GEMM prototypes xr3 / g31 / g32 / g33 all DEAD vs layer g30 / cuBLAS.
- s45 (NOTES s45): re-audit, no untested lever on any attention shape; lane request to lead (COORD). 0 checks.
## Attention: H200 batch-1 full facts (s32 DIAG tool tools/s32/mkdiag.py; NOTES s32)
- Floors (pt trip + 20.5 MB random 128-B rows, all in flight, incl. launch): ~8.0 us d1 and v1; dense 7.4.
- Main: d1 full 12.2-13.0, v1 full 17.0-17.3. With ALL gathers in L2: d1 11.0, v1 16.7 -> NOT memory-bound.
- Every plan knob loses (stages, splits 63-1023, BN, warps 8, pre/pf/fix, qk16, Gluon gw 127-263 +-gfuse, gbal).
- Next: H200 per-program timeline of _attn_split (v1) / _gd_full (d1) via the diag hook; build only once it names the cost.
## Attention: d1 full (s35; NOTES s35)
- ON MAIN 9855088: plan qf=True (query rows loaded before _gd_full's prologue copies): H200 -0.28 us (5/5 pairs, DIAGs
  95f8158e, 1259baad), H100 tie. Not yet in a scored run (rides with the next submit). Handoff to layer posted.
- H100 timeline (tools/s35/mktl_d1.py + run_d1.py): q 1.1, first tile +2.8, 1.17/tile (gather rate), red 0.45, _red_fin
  1.5 after the last program. H200: deeper stages, splits 640/1023/255, qf+640 all lose/tie. Unbuilt: 2-warp programs
  splitting each tile (per-warp chain 0.86 us/tile).
## Dead ends (older ones: NOTES-moe.md)
- (n5s8) pr32 at T=8 (g30n pooler + FP32 router FMA + routing, one launch) instead of pr8: H100 in-layer +1.8..4.3 us (pr8 already
  ~22 us after bnd in-layer); H200 L2W proxy 17.25 vs pr8 ~17: no win. PR_IL (router stages interleaved) worse.
- (n20, H200 13b528a8) LOSSLESS SCALE COMPRESSION (S6 6-bit codes, base 80; 5-bit windows would need escapes): loses at d16/d32
  (+1..2.7 us), wins only v1 -0.5..0.8: extra decode ops on a latency-bound consumer. Weight codes are uniform (4.0 bits): no gain there.
- (s26) L2 prefetch of the unpool W from twm (spread over its main phase, evict_last): v32 +5..10 H100 us, unpool -1.
- (s27) RSPLIT: weighted tk2f gemm13 split favouring CTAs that start early during pr1: +1.4..10 us (any imbalance costs).
- (s12, H200 diag on layer routings; REOPENED n4s5: its weighted split was slow + count split imbalanced) GS16 in any form (old, merged one-body tk2h, split weights 2-6): +4..8 us at d16/d32;
  per-segment gemm13->gemm2 counters (PSC): 0; deeper tk2f rings: +1..5; DYN: +3..10; NC16: +1..2. Weighted static
  split (TK2W): H100 worse, and the H200 timeline shows balanced CTAs (nothing to fix).
- (s10) GS16 16-token segments as default (judge bs32 135.7 vs 128.7; H100 layer d16/d32 +5..18 us); weighted row split for
  2-group rows (worse at every weight); ld.global.cg / asm nc loads for inter (no help: the race was the ring).
- (s9) Post-expert norm fused into tk2f (last-M-CTA tickets, one-pass register norm): judge bs1 +2 us (168fa04c).
  Also s7's grid-barrier version. A separate norm2 kernel is as cheap as any in-kernel tail on the H200.
- n7s23: tk2f XPF (gemm2 rows L2-prefetched at phase-1 end) and WAVE (two-wave row order to hide the phase barrier): H200 +0.7..5.9 us (DIAG 798fd574 / 87ee0bc0); the transition is not HBM idle.
