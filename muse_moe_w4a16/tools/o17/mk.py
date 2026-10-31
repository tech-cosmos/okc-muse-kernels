"""o17: write a submission copy with HOT_ET (layer v171: moe_hot triggers moe_twm before its own wait; twm polls meta[67]
bit 30 set by moe_hot CTA 0 after its wait). argv: src out_dir fix(0/1) delay_cycles.
fix 1 = moe_routegather CTA 0 zeroes meta[67] + __threadfence BEFORE its PDL trigger (race-safe: a stale/garbage bit 30
from the previous call can never be seen by this call's twm). delay > 0 = TEST ONLY: routegather CTA 0 spins that many
cycles after its trigger, before writing meta (widens the race window)."""
import sys, os
src, out, fix, delay = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = open(src).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
# moe_hot
rep("""  __shared__ int s_e[32], s_off[33], s_item[NSTG], s_nh;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];""", """  __shared__ int s_e[32], s_off[33], s_item[NSTG], s_nh;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  // o17 HOT_ET (layer v171): with per > 0 moe_twm (hx) launches now and polls meta[67] bit 30 (set below, after the wait)
  if ((per & ~(1 << 30)) > 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (blockIdx.x == 0 && tid == 0) { __threadfence(); atomicOr(meta + 67, 1 << 30); }  // item counter: & 0xFFFFF
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];""")
# moe_twm hx: poll instead of no wait
rep("""  if (!hx) asm volatile("griddepcontrol.wait;" ::: "memory");
#if HOT_EMU
  // s29 H200-proxy only""", """  if (!hx) asm volatile("griddepcontrol.wait;" ::: "memory");
  else {
    // o17 HOT_ET: moe_hot triggered this launch before its own wait: moe_routegather's output (meta, sroutes, xg) is
    // complete once moe_hot CTA 0 sets meta[67] bit 30 (after that wait)
    if (tid == 0) {
      int v;
      do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(meta + 67) : "memory"); } while (!(v & (1 << 30)));
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    __syncthreads();
  }
#if HOT_EMU
  // s29 H200-proxy only""")
# moe_routegather
z = """    if (tok == 0 && tid == 0) { meta[67] = 0; __threadfence(); }  // o17: before the trigger (HOT_ET's flag word)
""" if fix else ""
d = f"""  if ({delay} > 0 && tok == 0) {{ const long long t0 = clock64(); while (clock64() - t0 < {delay}) {{}} }}  // o17 TEST ONLY
""" if delay else ""
rep("""  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // n5s23: the x row loads""", """  asm volatile("griddepcontrol.wait;" ::: "memory");
""" + z + """  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
""" + d + """  // n5s23: the x row loads""")
os.makedirs(out, exist_ok=True)
open(out + "/submission.py", "w").write(s)
print("wrote", out, "fix", fix, "delay", delay)
