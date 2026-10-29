# STATE-fresh (newest first)
## f1s1 (2026-10-29 19:20Z-19:55Z): no layer build; 0 judge checks, 0 Modal
- Wrote muse_decode_layer/FRESH.md: per-load byte budget at 4.3 TB/s vs ours (v168 a5b4ddab), where the excess sits,
  3 hypotheses + oracles, critical review of the record. Ranking: H1 chain hops (d1/v1/d16/d32), H2 big-M experts consumer
  (v16/v32, moe's domain, ~+7% prize if at stream rate), H3 verify-full attention (dropped).
- Key review point: proj n7s4's 'piece free' oracle removed bytes but kept every kernel and wait, so its -2..-4 us says the
  d16/d32 pre-expert excess is latency, not bytes; it didn't test the persistent design. Many latency verdicts are H100-only.
- Started O1 (tools/fresh/chain_oracle.cu, H100 only): synthetic T=1 chain at real bytes. PRELIMINARY: persistent = PDL
  kernel-per-phase with prefill (121 vs 120 us NOMATH); per-hop cost = dependency wait + post-wait activation fetch
  (~3-8 us/hop), not the kernel boundary. The synthetic is ~55 us over bytes, so not yet a floor. Details: FRESH.md 'O1'.
- Then isolated one hop (tools/fresh/hop.cu, H100): counter 0.5 us + 32 KB x 1.5 us idle; HBM saturates at ~4.3 MB in
  flight, and each extra MB adds ~0.7 us of hop latency (bufferbloat = proj s39's finding, so not new). The chain at shallow rings:
  persistent still = PDL. H1 "persistent megakernel" NOT supported on the H100; the 17 us x-build was in-flight queueing.
- H200 (Modal 0.3 GPU-min): knee ~6.5 MB in flight; persistent = PDL (101.6 vs 101.3 NOMATH). H1 CLOSED. FRESH.md 'O1 on the H200'.
## Next (f1s2)
1. O2: size and locate the v16/v32 excess on the H200, in-layer: twm stream-only / dq-free / MMA-free (find moe's DIAG
   switches in moe_src.py first), plus per-kernel rooflines (dense GEMMs at M=128/256 look ~25 us over a good-GEMM rate at v32).
2. Then pick ONE build for session 3 and post it as `fresh -> lead` in COORD Requests.
2. Modal H200 (after 00:00Z; my cap 20 GPU-min/day; shared ledger was 50.3/60 on 10-29): chain + hop micro.
3. If hop >= ~2 us under load: design = fewer hops (redundant norms/routing in consumers), list the layer's hops per load.
   Post the table + pick under COORD Requests as `fresh -> lead`.
## Budget 10-29: layer checks 0/6 used, Modal 0.3/20 (shared ledger 51/60).
