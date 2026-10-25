# Overnight kernel work: Muse Spark Open, all decode-step problems

You are one of three Claude agents running unattended in a Daytona sandbox with one NVIDIA H100 80GB
(Hopper, like the judge's H200), working on the Open Kernel Challenge for the account tech-cosmos.
Each session is started by `tools/night_agent.sh <name>`. **Your lane is in `AGENT-<name>.md`**
(name given in the session prompt): read it after this file; it overrides this file where they differ.

## Why we're here (read once)

Prize money goes only to the Serving track (opens 2026-11-05). The top 50 on the **overall kernel
board** when the kernel challenge closes (**2026-11-05 08:00 UTC**) get priority there. The overall
board ranks by how much our best runs on the four decode-step problems together speed up a decode
step. SGLang's per-step times: muse_decode_layer 22,620 us (51%), muse_moe_w4a16 13,449 us (30%),
muse_bf16_proj 4,831 us (11%), muse_attention_decode 3,766 us (8%). On 2026-10-25 we were #815 of
1,129 overall: we had solved only attention (#9 there, 2.54x); 610 teams have solved all four.
So: **(1) get a correct run onto each board fast, (2) then make the big ones fast.** Real speedups only.

## Agents and lanes

| name | problem (dir in this repo) | notes |
|---|---|---|
| layer | muse_decode_layer | half the decode step; reuses the others' kernels |
| moe | muse_moe_w4a16 | NVFP4 W4A16 grouped GEMV/GEMM, skewed routing |
| proj | muse_bf16_proj | BF16 GEMV/GEMM, m = 1..256; also hands kernels to layer |

Attention (muse_attention_decode) is done by earlier agents: its code is in `~/attn`
(github tech-cosmos/okc-muse-attention-decode, read-only for you, see its STATE.md and NOTES.md).
The lead agent (the human's) runs its remaining checks; don't spend them.

Each agent works only in its problem's directory (plus its own notes) in its own clone
(`~/r-<name>`), on `main`. Commit small, `git pull --rebase` then `git push` after each commit; never
force-push or rewrite history. Never edit another agent's directory: if you want something changed
there, ask under "Requests" in ~/COORD.md. You may copy another agent's kernel into your own dir.

## Rules (non-negotiable)

- Follow the problem's TASK.md and AGENTS.md rules and the spirit of the challenge: CUDA-graph
  capturable, no host syncs, no caching anything derived from `data`, no allocating/branching on
  input values (only on shapes and `config`), weights at their own precision (NVFP4 stays NVFP4,
  BF16 stays BF16; repacking/lossless compression in `prepare()` is fine; dequantizing inside
  kernels is fine), activations BF16 wherever SGLang keeps BF16, FP8 KV cache stays FP8, queries no
  coarser than E4M3, attend to every key, no tampering with timing or harness, no work outside the
  timed call, no processes or network from the submission. Each run has 10-15 minutes including
  compile: keep JIT/compile fast and robust.
- Precision: never below SGLang's. Partials and accumulators FP32 where SGLang's are; when in doubt,
  match SGLang. The human decided on 2026-10-24: no BF16 split partials.
- If something you did turns out to be reward-hacky, `okc withdraw <job>` and say so in your notes.
- Do not touch anything outside your clone, ~/COORD.md, ~/okc (the venv) and /tmp. No new
  credentials, no other services. The okc rules were accepted by the human; agents never run
  `okc login`.

## Budget (per problem, per UTC day; resets 00:00 UTC)

Each problem has its own **20 checks and 10 scored runs** a day (`okc problems --json`). Your lane's
problem is yours alone:
- `okc check <dir>/submission.py --json`: use them all if you have real candidates. The judge
  reports per-shape times: one check can test independent changes on different shapes.
- `okc submit <dir>/submission.py --json`: at most 6 a day, only when your local correctness tests
  pass AND a check on the same code beat our best board result for the problem. **Exception: the
  first correct submission** (no row on the board yet) may be submitted straight away.
- Never resubmit unchanged code hoping for a lucky run.
- Log every check/submit (job id, commit, speedup, per-shape notes) in your NOTES-<name>.md and
  under "Judge runs" in ~/COORD.md.
- Out of checks is not a reason to stop: prototype on the H100 and queue candidates in STATE-<name>.md.

## Coordinating (~/COORD.md, shared, outside the repos)

- "Human decisions" at the TOP: the lead agent's direction. Read it IN FULL at session start;
  newest entries outrank everything else except the rules.
- "Claims": one line `<name> <UTC time>: <what this session tries>`; delete it at session end.
- "Judge runs": `<name> <UTC time> <check|submit> <job> <commit> <speedup>`.
- "Handoffs": a kernel another agent can reuse (`<name>: <what>, <path>@<commit>, <measured>`).
  layer should watch this: proj's GEMMs and moe's experts drop into the layer.
- "Requests": asks to another agent. "Results": one line per session (what, numbers, dead ends).
- Re-read the file right before writing it, change only your own lines, keep it short.

## Sharing the GPU

Three agents share one H100. Anything you time must hold the lock `/tmp/okc-gpu.lock`
(`fcntl.flock`, exclusive) only around the timed loop itself, never around compiles or tests. Keep
sweeps short. Don't kill other agents' processes. GPU memory is shared: free big tensors, and don't
hold more than ~20 GB.

## Environment

- `. ~/okc/bin/activate` (torch 2.14.1+cu130, triton 3.8.0, the judge image's versions). `okc` is on
  PATH, authenticated through `OKC_TOKEN`. CUDA 13.0 at /usr/local/cuda-13.0 for `load_inline`.
- The judge image `lmsysorg/sglang:dev-muse-spark-open` has SGLang (commit a55ab7d5), sgl-kernel,
  FlashInfer, DeepGEMM, TileLang, CuTe DSL and CUTLASS. **Submissions may import them** (SGLang's
  own kernels, e.g. Marlin MoE, FA3, are fair to call; beating them is the point). The sandbox may
  not have them: see `~/sglang-env.md` if the lead agent wrote one; otherwise don't spend more than
  ~30 minutes installing. Anything you install goes into ~/okc only, at the image's versions.
- No SGLang locally means baseline.py may not run here; the judge reports SGLang's per-shape times
  in every check, so write them into your STATE file the first time you see them.
- Build a local harness in your dir early (`bench.py`: correctness against reference.py on the
  test shapes, and CUDA-graph replay timing of the scored shapes with a cold L2, median of many
  replays). `~/attn/tools/bench.py` is a working example for the attention problem.

## Workflow

1. Read STATE-<name>.md (create it if missing: <= 150 lines; best board run with per-shape H200
   times vs SGLang, what's on main vs checked, queued candidates, ranked ideas, dead ends) and
   grep NOTES-<name>.md (the long log; newest at the bottom).
2. Check the board: `okc leaderboard <problem>` and `okc leaderboard` (overall).
3. One focused cycle (45-90 minutes): profile, change, test, time, check/submit per the budget.
4. Commit with measured numbers in the message; update STATE-/NOTES-<name>.md; push; stop.
   The next session starts fresh from your notes: write for a reader with no other context.
5. If `~/STOP` or `~/STOP-<name>` exists, wrap up and stop.
