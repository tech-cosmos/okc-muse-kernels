# fresh x42 (2026-11-02 05:40-06:30Z): routing slack ON vs OFF on v189, MEASUREMENT ONLY (lead 03:45Z). 0 judge runs.
Branch fresh-slack-h200 = v189 b24b7cb + layer/slack1 1fc464c (PR1_SLK at T=1) + moe/slack a9242fb (slack kernel at
T=128/256), margins unchanged (SLACK_MR/MK 0.023/0.022, SLACK1_TIE 0.02). OFF/ON files: mk_files.sh (TUNE pinned).
ON never touches d16/d32/v1 (T=16/32/8: no slack); PR1_SLK only d1; the slack kernel only v16/v32.

## H200 (Modal, h200_time.txt): 6 router seeds (11,23,37,41,53,67) x 4 inputs x 3 rounds, paired (on - off) mean +- se, us
| shape | off | on | delta |       | shape | off | on | delta |
| d1 sl | 65.75 | 63.94 | -1.81 +- 0.61 | | d1 fu | 73.06 | 71.78 | -1.29 +- 0.56 |
| d16 sl | 135.75 | 135.87 | +0.12 +- 0.11 | | d16 fu | 210.52 | 210.71 | +0.19 +- 0.13 |
| d32 sl | 157.83 | 158.03 | +0.20 +- 0.11 | | d32 fu | 304.98 | 305.10 | +0.13 +- 0.09 |
| v1 sl | 116.71 | 116.69 | -0.02 +- 0.10 | | v1 fu | 124.59 | 124.53 | -0.06 +- 0.07 |
| v16 sl | 274.87 | 274.61 | -0.26 +- 2.80 | | v16 fu | 368.17 | 369.58 | +1.41 +- 1.45 |
| v32 sl | 339.03 | 341.06 | +2.03 +- 0.60 | | v32 fu | 518.79 | 522.64 | +3.85 +- 0.90 |
Load gains (46 sl + 16 fu): d1 +2.51%, d16 -0.09, d32 -0.09, v1 +0.03, v16 -0.06, v32 -0.65 -> score +0.28% (all ON).
Untouched loads (d16/d32/v1) = +-0.1%: the noise floor. T=1 only (PR1_SLK, no slack kernel) = d1 alone: ~+0.42% score.
The T=128/256 slack kernel is a tie at v16 and a LOSS at v32 on the H200 (its ~4-9 us cost > ~0.85-1 expert saved).
H100 (h100_ni4.txt, same plan): d1 +1.27%, v16 +1.88%, v32 +0.42% -> +0.56%; the H100 overstates v16/v32.

## H200 routing stress (h200_judge.txt): slack ON, captured graph, fresh inputs, L2 flushed, check_routing vs ref_front
6 router seeds x 6 inputs x {d1, v16, v32} x {sl, fu} = 216 inputs: 0 FAILS. Worst pick gap 0.0272 (limit 0.05), worst
dropped-keep lead 0.0093 (limit 0.03). Distinct local experts saved per input: d1 ~0.29, v16 ~1.0, v32 ~0.85.
(H100: 0 / 240 too, h100_full.txt.)
