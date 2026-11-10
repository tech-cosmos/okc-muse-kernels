# okc-muse-kernels

Our entries for the four decode-step problems of the Open Kernel Challenge (Muse Spark Open), each
timed against SGLang's kernel on an H200 under CUDA-graph replay. The earlier, attention-only work
is in [okc-muse-attention-decode](https://github.com/tech-cosmos/okc-muse-attention-decode).

Almost all of the code and notes here were written by Claude Code agents running unattended in an
H100 sandbox: one agent per problem (`layer`, `moe`, `proj`), later a fourth (`fresh`), steered by a
lead agent through `CLAUDE.md`, the `AGENT-*.md` lane files and a shared coordination file. The
repo is published as it was when the sandbox was stopped on 2026-11-02, including the agents'
logs and the dead ends.

## Results

Best scored runs for the account `tech-cosmos` when work stopped (speedup over SGLang):

| problem | directory | best run | speedup |
|---------|-----------|----------|--------:|
| whole decode layer | `muse_decode_layer/` | `fcb67ee9` (v185) | 2.2048x |
| MoE, NVFP4 W4A16 | `muse_moe_w4a16/` | `5ef66b6c` | 2.3391x |
| BF16 projection | `muse_bf16_proj/` | `813c5e7a` | 1.3863x |
| attention decode | `muse_attention_decode/` | | 2.5962x |

`main` holds layer v189, which is later than the best scored run (v185) and was never submitted.

## Layout

| path | what |
|------|------|
| `muse_*/submission.py` | the kernel for each problem |
| `muse_*/TASK.md`, `AGENTS.md`, `baseline.py`, `reference.py`, `checks.py` | the challenge's own spec, baseline and checker, as written by `okc init` |
| `muse_decode_layer/build.py`, `*_src.cu`, `*_src.py` | the layer's sources; `build.py` inlines them into the single-file `submission.py` |
| `muse_*/dev/`, `muse_*/tools/` | per-session probes and microbenchmarks |
| `STATE-*.md`, `NOTES-*.md` | each agent's short current picture and long session log |
| `REDESIGN.md` | the MoE expert-kernel redesign plan |
| `CLAUDE.md`, `AGENT-*.md`, `COORD-template.md` | the instructions the agents ran under |
| `tools/` | sandbox setup, the agent loop and supervisor, H200 runs on Modal, score helper |

Branches (`layer/*`, `moe/*`, `proj/*`, `fresh*`) hold candidates and experiments, most of which
were measured and lost. The `*/slack` branches exploit tolerance in the routing check; they were
parked pending the organizers' approval and were never checked or submitted.

## Rules we held to

Nothing below SGLang's precision: FP32 accumulation everywhere, the KV cache stays FP8, weights
stay NVFP4. Decisions depend only on shapes and config, never on input values, and all work happens
inside the timed call.

## Running

Needs a Hopper GPU, torch 2.14.1+cu130, triton 3.8.0 and CUDA 13.0 (see `tools/setup_sandbox.sh`).
`baseline.py` needs SGLang, which the sandbox did not have.

## License and third-party code

MIT for our code (see `LICENSE`).

- `muse_decode_layer/vjx_src.py`, and the copy of it inlined into `muse_decode_layer/submission.py`,
  is [vjeux's](https://github.com/vjeux/okc-2026) full-layer attention kernel, used under the
  Apache License 2.0. The header of that file gives the source commit, our modifications and the
  full license text. `vjx_qlo.py` and `vjx_dyn.py` are our patches applied on top of it.
- The challenge files listed above belong to the Open Kernel Challenge organizers.
