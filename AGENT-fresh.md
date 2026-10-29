# Agent "fresh": fresh eyes on muse_decode_layer (started 2026-10-29 ~19:00Z by the lead, human's request)

You are a NEW agent. Three agents (layer, moe, proj) have worked on this layer for ~5 days and are now out of
ideas: every item they can name measures <= ~0.5%. Your value is that you are NOT anchored on their conclusions.

## Goal
#1 on muse_decode_layer (the overall board is ordered by it). Board: we are #63 at 2.158x (that run drew a slow
SGLang; our true level is ~2.14, see tools/fixed_sglang_score.py); #50 = 2.184x; #1 = 2.418x (mpeng19). A #1 team
spends ~12% less time per load than we do. Kernels close 2026-11-05 08:00 UTC.

## Where you work
- Your clone is ~/r-fresh on branch `fresh` (a copy of main = layer v171). You may edit anything under
  muse_decode_layer/ in YOUR clone (the layer agent is paused). Commit small; `git push origin fresh`. Never push
  to main; when you have a verified win, post a Handoff in ~/COORD.md and the lead merges it.
- Layer judge checks: up to 6 per UTC day for you (`okc check muse_decode_layer submission.py --json` in
  muse_decode_layer/). Log each one under "Judge runs" in ~/COORD.md as `fresh <UTC> check <job> <commit> <x>`,
  plus its fixed-SGLang score. Submit only if the check beats 2.158 and the lead agrees in COORD (the lead
  watches every ~30-60 min).
- Modal H200 (tools/h200.py, daily cap in ~/.h200_ledger.json): up to 20 GPU-min per UTC day for you.
- H100 timing: tools/bench.py / ab_files etc. take /tmp/okc-gpu.lock; keep sweeps short.

## How to work (the important part)
Session 1-2, NO BUILD: build your own model of the problem before reading anyone's conclusions.
1. Read muse_decode_layer/TASK.md, reference.py, checks.py, and the structure of the current submission
   (layer.py + build.py generate submission.py; skim, don't study every kernel).
2. Read the data, not the notes: judge profiles in muse_decode_layer/profiles/*.json|txt (per-shape medians,
   per-kernel rows), the H200 per-kernel timeline muse_decode_layer/dev/n7s1_tl_h200.txt.
3. Write FRESH.md (in your clone, muse_decode_layer/FRESH.md): per load (d1/d16/d32/v1/v16/v32, sliding and full),
   (a) the bytes and FLOPs that must move, (b) the time an ideal H200 implementation would take per segment
   (qkv, attention, o, glue, router/pooler, experts, tail), (c) our time per segment, (d) what a 2.42x team must
   have per load. Then your top 3 hypotheses for HOW the top teams are faster, each with a cheap ORACLE test
   that would confirm or kill it (an oracle = a measurement of the best case, e.g. a piece made free or its
   weights pre-warmed, on the H200).
4. Only THEN check each hypothesis against the record: STATE-layer.md / STATE-moe.md / STATE-proj.md dead-end
   lists, then grep NOTES-*.md. For anything already 'measured dead', read the record critically: was it
   measured on the H100 instead of the H200, in a microbench instead of the layer, in one narrow variant, or
   with an assumption you think is wrong? Re-open it only if you can name the flaw. Write that review into FRESH.md.
5. Run your oracles (Modal H200 preferred: the H100 often misleads here), post the table + your pick under
   "Requests" in ~/COORD.md as `fresh -> lead (...)`, then build the winner from session 3 on.

## Facts you may rely on (measured on the H200 unless noted)
- Our standalone expert kernels already beat the best public ones (vLLM Humming 1.4-1.7x slower; CUTLASS
  mixed-input slower). The moe-board #1 is still 18% faster than us there, so a better design exists unpublished.
- In-layer experts are ~45-60% of layer time; on d16/d32 they stream at ~4.33 TB/s marginal per expert.
- Judge noise: SGLang's per-shape times swing 2-6% between runs, ours ~2%; one check can't resolve < ~2%.
- Rules (CLAUDE.md, TASK.md) are non-negotiable. Choosing routes among near-ties to drop experts is OUT
  (checker slack, not speed). Precision never below SGLang's; FP32 partials; BF16 activations; FP8 KV.
Report each session's result as one line under "Results" in ~/COORD.md prefixed `- fresh fNsK (...)`.
