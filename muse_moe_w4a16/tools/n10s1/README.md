# n10s1 (2026-10-30): layer END lane (tk2f end -> norm2 -> unpool -> bnd), H100 oracles
Run from a layer dir (muse_decode_layer/, needs bench.py). GPU lock is taken around timed loops.
- tl4.py <sub.py> [reps]: per-kernel start/end relative to moe_tk2f's END, d1/d16/d32/v1 sliding (median of cold replays).
- mkvar.py <sub.py> <outdir>: oracle variants (TUNE = {}): nt_nobnd, nt_hop, nt_nfree, nt_q4 (see docstring).
  Then: python tools/ab_files.py sliding 6 outdir/nt_base.py outdir/nt_hop.py ... (ALTERNATE the order across
  repeats: separate module instances carry a ~0.5 us offset; file position bias was visible on d1/v1 controls).
- bndcap.py <sub.py>: pickles the second _boundary's args per T to /tmp/moeend/bnd_args.pt;
  bndmicro.py / micro2.py: chained (20 PDL launches) per-launch time of Triton _boundary vs bnd2_src.cu (CUDA rewrite;
  micro2 reads /tmp/moeend/cb/bnd2_src.cu: copy it there). bndasm.py: PTX load/wait order of the Triton boundary.
- PTXAS RULE (again): ptxas hoists griddepcontrol.wait (ACQBULK) above plain loads issued "before" it, and hoists .nc
  loads of producer data whose address is known ABOVE it. Pin pre-wait loads with a branch on the loaded words (a
  volatile store to a private smem word), load producer data with __ldcg. Check: cuobjdump -sass | LDG vs ACQBULK.
