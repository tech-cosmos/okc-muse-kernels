# STATE-fresh (newest first)
## x40 (2026-11-01 00:09Z-00:15Z): status only, 0 judge, 0 Modal, 0 GPU. Lead 23:55Z items (a) CSL_PW H200 A/B and (b) x24 vjeux
  A/B were both done in x39b/x39c. No newer lead line; board #185 at 2.205 (#1 pjmullin3 2.469). Idle-stop per lead 23:55Z.
## Next (x41): read Human decisions + lead lines naming fresh (human checks in 11-01; vjeux copy + QLO full half are human calls).
  Checks 11-01: 0/6; Modal 11-01 7.5/20.
## x39 (2026-10-31 23:03Z-11-01 00:10Z): lead 22:55Z WHOLE-LAYER HOP STAMPS d16 sl + d32 sl (H100, v185 pinned), 0 judge, 3.7 Modal.
  New tool tools/fresh/x39/lstamp.py (load_inline hook: per-warp entry/exit + pre/post griddepcontrol.wait stamps in every CUDA
  kernel; Triton kernels too) + lan.py / cta.py / strag.py. FRESH.md x39 has both tables. PDL releases 0.4-0.8 us except
  bnd->pooler 2.7-3.2 and d16 qkv->csl 3.2 (consumer pre-wait bursts in flight; free-boundary bound 1.5/1.9 on record);
  pooler end->tk2f release 6.9/5.8 us (all pieces on record; tk2f can't co-reside with the pooler); unpool late CTAs (blocked
  by norm2) = tail, bounded by proj n7s1 R2 (-0.5/-0.9 H200). Nothing >= 1 us unbounded -> nothing built.
  H200 A/B CSL_PW T=1 (v185's change): d1 sl -1.3 us (69.74/69.50 -> 68.53/68.16, ABBA). Confirmed.
  x24 H200 attention A/B (measurement only): vjeux faster on all 6 FULL shapes (-0.9..-9.3 us, v32 fu 193.2 -> 183.8) and d32 sl;
  ours faster d1/v1 sl. ~+0.6% layer if carried 1:1. Human decides on copying (FRESH.md x39). Modal 11-01: 7.5/20.
## Next (x40): read Human decisions + lead lines naming fresh. Candidates if asked: the d16 sl attention phase (qkv end -> csl
  end 13.6 us H100 for ~8 MB of page-1 gathers; csl CTAs reach their wait 7.7 us after entry) for the attention owner; v16/v32
  whole-layer stamps with lstamp.py (STP_SKIP=layer_g27_ if g27 is in the path; pin TUNE). Checks 11-01: 0/6; Modal 11-01 7.5/20. Lead 23:55Z: idle-stop OK when nothing is open.
## x38 (2026-10-31 22:47Z-22:52Z): status only, 0 judge, 0 Modal, 0 GPU. Board #180 at 2.205 (layer v185 fcb67ee9 = v183 + fresh
  x35 CSL_PW T=1); #1 pjmullin3 2.469. No lead line since 21:50Z; lane request (x36) open.
## Next (x39): after 00:00Z run tools/fresh/x35/mk_h200.sh (H200 A/B CSL_PW T=1 on d1 sl, ~3-4 GPU-min) as confirmation of what
  v185 already scored; report under Results; then act on a lead line naming fresh, else status only. Modal 10-31 16.8/20.
## x37 (2026-10-31 22:31Z-22:40Z): status only, 0 judge, 0 Modal, 0 GPU. Board #217 at 2.187 (#1 pjmullin3 2.459). No lead line
  since 21:50Z. x36 (90da00a) closed the d1 sl csl->o flag handoff (+1.1..+2.5 us, DEAD) and the attention lane.
## Next (x38): after 00:00Z run tools/fresh/x35/mk_h200.sh (H200 A/B CSL_PW T=1 on d1 sl, ~3-4 GPU-min); report under Results;
  then act on a lead line naming fresh, else status only. Checks 10-31: 4/6 (resets 00:00Z); Modal 10-31 16.8/20.
## x35 (2026-10-31 21:47Z-): lead 21:50Z stamps of d1 sl / v1 sl attention in-layer (H100; FRESH.md x35, tools/fresh/x35/astamp.py).
  d1 hops: qkv end -> csl q ready 2.7 us (launch 1.3 + first loads 1.2), KV 0.3, QK/PV 0.55, cluster 0.35, merge->store 0.9,
  store -> o g13 has x 0.85. v1: vtk first loads 1.8, wait 1.2 after g14's last CTA, merge->store 1.0; vtk can't start early
  (16-CTA clusters don't fit beside g14). BUILT (branch fresh f064596): CSL_PW_TOKENS (16, 1) + csl smem carveout 100 only for
  the T=16 instance: s62's d1 loss WAS the carveout. H100 d1 sl -1.2..-1.4 us (ab_files mirrored, pinned), d1 bitwise vs
  v183, pw_race d1 0/20, validate ALL OK (prep 60.8 s). d16 keeps the carveout (removing it +0.8 us). Judge check: see below.
  Check 4a3434a9: ok, raw 2.1959 / fixed 2.1629 (v183 2.1576), d1 sl 64.4 (v181-184: 65.2-67.8). Handoff posted. 4/6 checks 10-31.
## Next (x36): Human decisions + lead lines. After 00:00Z: tools/fresh/x35/mk_h200.sh (H200 A/B d1 sl, ~3-4 GPU-min of my 20).
## x34 (2026-10-31 21:31Z-21:55Z): lead 21:25Z question answered: v174 (c62cdb0 submission.py = 50534b24) CRASHES on non-SA
  weights (layer tools/n37/nonsa.py, H100: tuner skips all, first test raises 'MOE_PRUNE: tk2f cfg 84204 not built'). All v157..v182
  have it; generator weights are always SA. Posted under COORD Requests. 0 judge, 0 Modal. Next (x35): after 00:00Z, x24 H200
  attention A/B (tools/fresh/x24/mk_h200_attn.sh) as MEASUREMENT only (lead 21:25Z; no vjeux code in the build). Checks 10-31: 3/6.
## x33 (2026-10-31 21:15Z-21:20Z): status only, 0 judge, 0 Modal, 0 GPU. Board #224 at 2.172 (#1 pjmullin3 2.459). No lead line
  since 16:30Z; no STOP file. Next (x34): read Human decisions + lead lines naming fresh; else status only. Checks 10-31: 3/6; Modal 16.8/20.
## x32 (2026-10-31 21:03Z-21:30Z): HELD-OUT WEIGHT SEEDS for v184 (70591ea, submission sha1 926ce2ec = layer's 00:00Z submit), 0 judge,
  0 Modal. /tmp copies with reference.WEIGHTS_SEED 31337 and 2 (never tested; proj n26 did v181 on 1/777), validate --quick
  --skip=prep,fuzz: BOTH ALL OK (test 13/13, scored 0/12, opt 0/40, tk2f 0/300, replay 0/240, route 0/240; worst gap 0.0275/0.0214
  (limit 0.05), drop-lead 0.0145/0.0187 (limit 0.03)). v184 is safe on held-out weights. Next (x33): Human decisions + lead lines; else status only.
## x31 (2026-10-31 20:48Z-20:55Z): status only, 0 judge, 0 Modal, 0 GPU. Board #223 at 2.172 (#1 pjmullin3 now 2.459). No lead
  line since 16:30Z. Next (x32): read Human decisions IN FULL + lead lines naming fresh; else status only. Checks 10-31: 3/6; Modal 16.8/20.
## x30 (2026-10-31 20:31Z-20:45Z): judge-data audit of the v182-v184 'slow d1/v1' (0 judge, 0 Modal, 0 GPU). NODE, not code:
  v1 sl (idx 6) is bimodal with SGLang moving along: fast 108/261 us (v181 84eb0719, v178 9117bd11), slow 115-119/274-279
  (v179 36474b5e = pre-fix code, v182, v183, v184). d1 sl 64.1-67.8 us with SGLang flat 171-172 = noise band. Backs layer n40's
  v184 submit at 00:00Z. Next (x31): read Human decisions + lead lines naming fresh; else status only.
## x28 (2026-10-31 20:00Z-20:05Z): status only, 0 judge, 0 Modal, 0 GPU. Board #222 at 2.172 (#1 mpeng19 2.418), unchanged.
  No lead line since 16:30Z; fresh's Requests (x27: v182/v183 submit pick, lane or STOP-fresh) still open. layer n37 now
  recommends v183 (non-SA fix) for 00:00Z; fresh agrees (v183 > v182 > v181 for held-out robustness, same speed).
## x27 (2026-10-31 19:42Z-19:55Z): paper, 0 judge, 0 Modal, 0 GPU. No new lever. Board: #222 at 2.172 (#1 2.418). Re-read the twm
  consumer record (f1s12, x1, moe superopt): no flaw found. COORD Requests: recommended v182 for the 00:00Z submit (held-out
  fallback safety, same speed), and asked the lead for a lane or STOP-fresh.
## Next (x28): read Human decisions IN FULL plus any `lead` line naming fresh. With no lead line: status only (don't spend
  budget). After 00:00Z, only on the lead's yes: tools/fresh/x24/mk_h200_attn.sh. Checks 10-31: 3 of 6 used; Modal 10-31 16.8/20.
## x26 (2026-10-31 19:26Z-19:40Z): tuner-pick noise audit (39 judge logs): candidates are near-ties (median gain vs default 0-1.2 us),
  so wrong picks cost <= ~1 us typical: CLOSED. 0 judge, 0 Modal, 0 GPU. Next as x25's Next (unchanged).
## x25 (2026-10-31 19:07Z-19:15Z): own pick, paper only (0 judge, 0 Modal, 0 GPU). FRESH.md x25. Nothing to build.
  PDL per-kernel rows can't be compared across teams (x24's "their d32 MoE 85 vs our 95+14" is wrong: moe_fused d32 sl is 86-100
  vs our tk2f 64.8). Attention-board #1's lead is probably sliding batch-1 latency; in the layer, the free-oracle cap is
  d1 2.4 / v1 3.3 us (<= 0.6%). Weakest loads d32 1.83-1.93 / v16 1.99-2.06 are already mapped in the record.
## Next (x26): read Human decisions IN FULL. After 00:00Z (Modal share resets to 20): tools/fresh/x24/mk_h200_attn.sh (~11 min,
  vjeux vs ours, H200 full shapes) if the lead still wants it; build only on a lead line + >= 5 us at v32 full.
  Checks 10-31: 3 of 6 used; Modal 10-31 16.8/20.
## x24 (2026-10-31 ~18:10Z-19:06Z): public OKC repos per shape (FRESH.md x24): no sliding lead anywhere; full d32/v1/v16/v32
  public code ~3-12 us ahead via vjeux's one-kernel attention (<= 0.2-0.4%); H100 standalone tie. 0 judge, 0 Modal.
## x23 (2026-10-31 18:52Z-19:10Z): own pick, COLD CODE after the L2 flush (H100, 0 judge, 0 Modal). FRESH.md x23.
  Twin-state oracle (tools/fresh/x23/codewarm.py: replay an identical twin first so all SASS is in L2, data stays cold):
  d1 sl -1.35, d1 full -0.34, v1 sl / d32 sl +0.7/+0.8 (twin's dirty lines) -> <= ~1.4 us d1 sl, unbuildable (device code
  can't read kernel code addresses: fp.cu illegal access). CLOSED. Full loads are not bitwise run-to-run (1-ulp order noise).
## Next (x24): read Human decisions IN FULL. Own list empty again. After 00:00Z (Modal share 20): the remaining H200 A/B
  (v16 sl / v32) only if layer asks. Checks 10-31: 3 of 6 used; Modal 10-31 16.8/20.
## x22 (2026-10-31 18:24Z-18:55Z): own pick, x21's last open item: T=128 qkv split-K g30 oracles (H100, 0 judge, 0 Modal). FRESH.md x22.
  Memory free (no-mem 20.5 = full 21.7 us, cuBLAS 20.6); split-K reduce ~5 us in ANY form (global slabs or DSMEM cluster
  reduce-scatter: one cross-CTA handshake ~2.2 us); best case ~18.9 -> <= 1.7 us on v16 qkv: DEAD. Dense lane closed (T=128+256).
## Next (x23): read Human decisions IN FULL. My own list is EMPTY (no item >= 0.5 us left that the record or x1-x22 haven't
  killed). Act on a lead line naming fresh; else after 00:00Z the remaining H200 A/B (v16 sl / v32) only if layer asks.
  Checks 10-31: 3 of 6 used; Modal 10-31 16.8/20.
## x21 (2026-10-31 17:35Z-18:25Z): own pick, T=256 dense GEMM headroom (H100, 0 judge, 0 Modal). FRESH.md x21, tools/fresh/x21.
  s100 T=256 g30 form (38.9 vs cuBLAS 28.7 us) is NOT memory-bound (no memory at all 37.0): in-loop escape fixup ~5 us, sync skeleton
  ~200 ns/stage, MMA loop 27.4. Pure RS wgmma loop floor 24-25 us (~700 TFLOPS) vs cuBLAS 28.9 -> T=256 dense lane DEAD (<= 3-4 us
  per GEMM before costs). Triton screen never beats nvjet (5 shapes). Only untried: T=128 qkv split-K with GX escapes (gate first).
## Next (x22): read Human decisions IN FULL. If the lead wants v16 qkv: tt2o-style oracles on dev/s101 split-K (no-fix/no-mem) as
  the gate; else idle until the lead names an item (my Modal share 10-31 is 16.8/20; checks 3 of 6 used).
## x20 (2026-10-31 17:21Z-17:40Z): own pick: re-measured the record's "per-SM ingest wall" (H100 + H200 microbench, 0.2 Modal min, 0 judge).
  It is the chip's DISTINCT-LINE L2 rate (7.5 TB/s H100 / 8.6 H200 = 57-66 GB/s/SM); SAME lines read concurrently by all SMs: 16.5 TB/s
  (125-159 GB/s/SM). One TMA producer thread: ~105 ns/stage floor (6 KB stages <= 60 GB/s/SM). Paper bounds built on the wall (proj s90,
  FRESH f1s2/f1s3) are invalid; measured kernels stand. Dense T=128/256 bound = HBM+tensor (~15-25 us/load over nvjet at v16/v32, ~1.5-2%)
  but needs a kernel that beats nvjet with x read in K-lockstep. FRESH.md x20, tools/fresh/x20. Modal share 16.8/20.
## Next (x21): read Human decisions IN FULL. If the lead wants the dense lane: an own T=256 qkv GEMM prototype (lockstep x, 12-bit W,
  16 KB+ stages) vs nvjet on the H100; else after 00:00Z the remaining v16 sl / v32 A/B if layer still wants it.
## x19 (2026-10-31 17:20Z-17:55Z): the planned H200 A/B run early (ledger had room; my share 16.6 of 20 used). 0 judge.
  v176 vs v177, tuner pinned off: d32 sl -4.3 us, d32 full -3.4 (ABBA, +-0.4), d16 sl tie, d16 full -0.9, v16 full tie.
  -> x14 (NDS_PW/NVS_PW/GD_PW/CB_HB_TRIG) + MRG 2 all KEEP on the H200 (already in main v177..v181). Not run: v16 sl, v32.
  Branch fresh = main v181 merged (624c913). tools/fresh/x16/mk_h200b.sh = mk_h200.sh with full output (LOADS='decode 32,...');
  the old one's `tail -12` kept only the last variant: don't use it.
## Next (x20): read Human decisions IN FULL. After 00:00Z (Modal share 20): v16 sl + v32 sl/full v176 vs v177 if layer still wants
  them (low value: x14 only touches nvs there); else own pick. Checks 10-31: 3 of 6 used.
## x18 (2026-10-31 16:37Z-17:05Z): lead 16:30Z 'nvs QLO into v178': main v178/v179 do NOT have it -> built the bundle.
  Branch fresh 892bb2a = main v179 (faf3590) merged + x16 nvs QLO (attn_src.py only; main's attn_src unchanged since v177,
  clean merge; build.py reproduces submission.py, 998541 B). bench test 13/13, replay_test verify 0/360 (seeds 11,12,13 x 20).
  Handoff to layer: merge = v180 for the post-00:00Z check/submit. 0 judge, 0 Modal.
## Next (x19): after 00:00Z run muse_decode_layer/tools/fresh/x16/mk_h200.sh (v176 vs v177 pinned; still answers x14 per shape).
  GW QLO waits for the human. Read Human decisions IN FULL first.
## x17 (2026-10-31 16:20Z-16:35Z): short stop, no lead line naming fresh after 15:58Z ('after 00:00Z the Modal A/B as planned').
  0 judge, 0 Modal. Searched for a cheaper GW (_gw_verify) QLO form: M = L*G = 64 rows, D 64 (no padded A rows/K to hide the lo
  term in); paying for it with FP8 P@V is dead on record (~/attn NOTES: slower on every verify shape, error up to 0.105). GW QLO
  stays a human decision at ~0.3%. Checks 10-31: 3 of 6 used; Modal 10-31 ledger 32.1 (all agents).
## Next (x18): after 00:00Z run muse_decode_layer/tools/fresh/x16/mk_h200.sh (unchanged plan; layer wants all 12 shapes if the
  20 GPU-min allow: add 'decode 1' / 'verify 1' only after the 4 groups). Read Human decisions IN FULL first.
## x16 (2026-10-31 15:51Z-16:25Z): own pick, ROUTING-FAIL RISK. 1 judge check (2d0914bf), 0 Modal. FRESH.md x16.
- FP8 (E4M3) query rounding is our whole router-score error (= the E4M3 reference's); a two-term query q = e4m3(q) +
  e4m3(q - e4m3(q)) cuts it 15x in the reference (worst drop-lead <= 0.0014 vs 0.02-0.025; limit 0.03) and 5x in our layer.
- Branch fresh 4725658 = v177 + QLO in nvs (v16/v32 sl): drop-lead 0.016-0.020 -> ~0.004, update error 6% -> 1.2%, H200 nvs
  +0.2-0.4 us. Test 13/13, replay verify 0/360. QLO in _gw_verify (v16/v32 full) costs +9% kernel on the H200 (+8.9/+17.3 us,
  ~0.3% score): dropped, patch kept (tools/fresh/x16/qlo_patch.py ... GW). Check 2d0914bf raw 2.1745, fixed 2.1405 (slow node).
- Checks 10-31: 3 of 6 used.
## Next (x17): after 00:00Z run muse_decode_layer/tools/fresh/x16/mk_h200.sh (Modal H200 A/B v176 vs v177, pinned tuner, 4
  load groups, ~14 GPU-min; layer's request; read the bench output per shape, no routing confound: numerics equal); d32 sl nds QLO only if
  the human wants the rest of the risk (FRESH.md x16).
## x15 (2026-10-31 15:32Z-16:20Z): lead 15:25Z 'continue'. 0 judge, 0 Modal. FRESH.md x15.
- CORRECTION: the x14 judge FAIL was on v32 sl REPETITION 1 with the run's own router seed; repro_v32.py never reproduced that
  input. NVS_PW code read: pre-wait producer touches inputs/old KV only (no race found). Record ~2 routing fails / ~100 layer
  runs. If layer's v177 check fails v32 sl again, suspect NVS_PW first (NVS_PW 0 in attn_src.py).
- H100 timelines d16/v1 sl on branch fresh: nothing new beyond the record (front chain, pooler window, first-op ramp).
## Next (x16): after 00:00Z Modal H200 A/B v176 vs v177 (or x14 / MRG 2 split if v177's check is ambiguous); d32 sl+full,
  v16 sl, d16 full; one load per Modal call. Read Human decisions IN FULL first.
## x14 (2026-10-31 15:00Z-): lead 15:00Z early-trigger audit. FOUND: the attention kernels after qkv were not PDL. FRESH.md x14.
- Built on branch fresh (attn_src.py only): NDS_PW (nds = qkv's PDL secondary, KV producer pre-wait, trigger after the
  consumer's wait), NVS_PW (same for nvs), GD_PW (batched _gd_full), CB_HB_TRIG on. H100 pinned-tuner A/B: d32 sl -3.1
  (ab_flag CB_HB_TRIG another -1.4/-1.6), v16 sl -0.7, GD_PW d16 full -2.2 (ab_flag) / tie (pinned); rest noise. Exact:
  test 13/13, graph_eq bitwise on sliding, replay 0 fails everywhere.
- Check 05077ac2 (main v176 + x14 attn_src.py) FAILED on a v32 sl routing tail draw (margin 0.0305 > 0.03; outputs
  identical to v176, same local margins): no score. 11 shapes: d32 sl 152.1 vs 154.6; fixed-SGLang (v32 imputed) 2.1962.
  Handoff posted (layer merges attn_src.py). Checks today: 2 of 6 used.
## Next (x15): after 00:00Z Modal H200 A/B v176 vs v176+x14 attn_src (d32 sl, v16 sl, d16 full; ~3 GPU-min) and the
  x13 MRG 2 A/B (d32). Untried same-pattern items: _attn_split (v1 full, ~0.6 us start gap, .033 %/us) - small.
- Rule learned: whole-file A/Bs with the tuner on compare different tuner picks; pin TUNE for A/B and graph_eq.
## x13 (2026-10-31 14:27Z-15:10Z): lead 14:30Z CW for T=8 / T=16/32. 1 judge check (9eeab096), 0 Modal. FRESH.md x13.
- T=8 CW built (segment-major CwIt, bitwise 300+300) but H100 v1 NEUTRAL; no-wait oracle v1 <= 0.5 us, d16 BAL no-wait
  -0.3/0: CW closed at T=8/16 (stream hides the transition). Patch tools/fresh/cw/cw8_sm_neutral.patch (not kept).
- d32 tk2h waits on ONE global gemm13 counter (MRG 1). MRG 2 (existing per-segment counters, one-line layer change):
  H100 d32 sl -2.7 / full -2.2 us (= no-wait oracle), test 13/13, replay 0/120, sl bitwise. Branch fresh 9325413 =
  main v176 + MRG 2. Check 9eeab096: raw 2.1736 / fixed 2.1460 (slow draw; d32 sl 157.8 vs v176 154.6): UNRESOLVED.
## Next (x14): read Human decisions IN FULL. After 00:00Z: Modal H200 A/B d32 sl+full, v176 vs v176+MRG 2 (~3 GPU-min,
  command in FRESH.md x13). Win >= 1.5 us sl -> Handoff to layer; else close MRG 2 (H200 balanced, as moe's PSC).
## x12 (2026-10-31 13:58Z-14:30Z): lead 13:58Z BUILD CW: DONE, WIN on the H100 (d1 sl -0.5, full -1.55 us, 4 seeds). 0 judge, 0 Modal.
- Branch fresh = v174 + CW v3 (layer TK2F_CW = True, cfg 184294 = tk2f PF 4 | 8192, T=1 only; patch tools/fresh/cw/cw3.patch).
  gemm13 rows interleaved per segment (i & 3) * 64 + i / 4, CTA order chunk-major; per-(segment, chunk) counters cnt[64 +
  seg * 4 + c] published by a helper warp (tk2f_cw_helper); gemm2 blocks on chunk 0 only, polls later chunks per stage.
- Bitwise 1500 eager + 1500 replay routings; test 13/13; replay_test 0 fails. Handoff posted (layer: bundle + check; moe:
  bs1 port). First permutation (v1, 16 v + u) was neutral: wrong for 5.8 rows/CTA (FRESH.md x12, correction).
## Next (x13): read Human decisions IN FULL. If asked: (a) H200 A/B of CW vs v174 on d1 (Modal after 00:00Z, ~3 min),
  (b) CW at T=8 (v1: up to 64 segments -> counter area + zeroing; nu8 PF 16 path), (c) help moe port CW to the moe board's
  M=1 tk2f (bs1 fixed cost: x11 O2 bound 1.4 us). Stamp tools: tools/fresh/cw/{stamp,strun}.py.
## x11 (2026-10-31 13:34Z-13:56Z): lead 12:25Z MOE BS1 GAP, cont. H100 stamps + oracles, 0 judge, 0 Modal. FRESH.md x11.
- bs1 fixed ~9.5 us = pre-data 4.3 (routing 1.5, TMA issue 0.5, first data 2.3) + w13->w2 transition 1.4 (O2) + norm
  2.4 (O1) + launch/PDL ~1. In-layer: transition bound ~1.1 us d1 / ~0.5-0.8 d16/d32 (O2); n2u work 3.5/4.4 d1 (O3,
  latency chain, every fix on record dead); boundary ~0 (O3b).
- Candidate CW (chunk-wait: q-major strided gemm13 rows + per-(expert, chunk) counters, gemm2 waits per stage):
  est. ~0.6 us d1, ~1 us bs1; NOT built (multi-hour inter-CTA protocol). Asked the lead in COORD Requests.
## Next (x12): read Human decisions IN FULL. If the lead says build CW: branch fresh, T=1 first (cfg 84204 path), the
  stress test (many routings, replay) before timing; H100 gate >= 0.5 us d1 sl via tools/ab_files.py base,c,c,base.
  After 00:00Z, the H200 A/B of the standalone 'late' patch if the lead/moe want it (Modal share 20 min).
## x10 (2026-10-31 13:12Z-13:40Z): lead 12:25Z MOE BS1 GAP. H100 fit bs1: t = 9.5 + 10.0 x E us (fixed 9.5 = the gap).
  Candidate 'late' (tools/fresh/bs1/late.patch on muse_moe_w4a16/submission.py: TK2F 84204 at M=1 + NORM2_PDL 1 + tk2f
  trigger at its gemm2 rows): tests 15/15, H100 every shape faster, geomean x1.846 -> x1.868. Handoff to moe/lead; needs
  H200 (moe n5s19: PDL norm2 lost +3.4 us d1 there with the ENTRY trigger). FRESH.md x10. 0 judge, 0 Modal.
