# moe/slack tools (moe s5, 2026-10-31). Routing-slack selection: NOT for submission until the organizers approve.
- slack_judge.py <file> <n> <router seeds> [filter]: graph-replay routing stress (route_judge.py copy) + distinct/routes saved
  vs the reference top 8. Result (margins in layer.py): 0 fails / 192 slack-on inputs (v16/v32, routers 11/23, 24 inputs).
- keyerr.py / leaderr.py: the layer's router keys vs the FP32 reference (eager, layer._SLACK_DBG hook). Keep-rule lead error
  on droppable experts: std ~0.0048 (max 0.019), = the reference with E4M3 queries vs FP32; ours vs the E4M3 reference 0.0025.
- tst_standalone.py <noise> <MR> <MK> <nseeds> [exact] [TIE] [DM] [RM]: slack_src.cu on reference logits, check_routing,
  saved distinct vs fresh x8's exact minimum (exact = any 5th arg). Margins 0: greedy == exact on all 12 loads.
- ktime.py: the slack kernel alone in a graph (T=128 ~4.4 us, T=256 ~8.7 us incl. node; small T 7-16 us, unexplained).
- ab_seeds.py <filter> <rounds> <seeds> A B: paired A/B over input seeds.
