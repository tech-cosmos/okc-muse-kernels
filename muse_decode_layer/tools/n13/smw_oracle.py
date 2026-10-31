"""n13s2: SM-weighted static split oracle (H100). Bare stream reader, 1 CTA per SM (big dyn smem), each CTA reads a
contiguous byte range of an MB-sized buffer with ld.global.v4 (U loads in flight/thread). Range per CTA comes from a
table indexed by %smid (mode table) -> equal split vs a split weighted by the measured per-SM rate (iterated).
Stamps per CTA: smid, start, end (%globaltimer). Cold L2 (flush write before every launch), events around the kernel.
usage: python tools/n13/smw_oracle.py [MB=28.3] [ITERS=3] [REPS=40]"""
import sys, fcntl, statistics as st
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ uint64_t gt(){uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;}
__device__ __forceinline__ unsigned smid(){unsigned s; asm volatile("mov.u32 %0, %%smid;":"=r"(s)); return s;}
template<int U>
__global__ void __launch_bounds__(512,1) rd(const uint4* __restrict__ p, const long long* __restrict__ rng,
                                           unsigned long long* st, float* out){
  extern __shared__ char sm[];
  unsigned s = smid();
  uint64_t t0 = gt();
  long long a = rng[s], b = rng[s+1];          // in uint4 units
  uint4 acc = make_uint4(0,0,0,0);
  long long i = a + threadIdx.x;
  for (; i + (U-1)*512 < b; i += U*512){
    uint4 v[U];
    #pragma unroll
    for (int u=0; u<U; u++) v[u] = __ldcs(p + i + u*512);
    #pragma unroll
    for (int u=0; u<U; u++){acc.x^=v[u].x; acc.y^=v[u].y; acc.z^=v[u].z; acc.w^=v[u].w;}
  }
  for (; i < b; i += 512){uint4 v=__ldcs(p+i); acc.x^=v.x; acc.y^=v.y; acc.z^=v.z; acc.w^=v.w;}
  __syncthreads();
  uint64_t t1 = gt();
  if ((acc.x^acc.y^acc.z^acc.w) == 0x12345677u) out[0] = 1.f;
  if (threadIdx.x == 0){ st[3*blockIdx.x]=s; st[3*blockIdx.x+1]=t0; st[3*blockIdx.x+2]=t1; sm[0]=0; }
}
void run(torch::Tensor p, torch::Tensor rng, torch::Tensor st, torch::Tensor out, int grid, int smem){
  auto k = rd<8>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  k<<<grid,512,smem>>>((const uint4*)p.data_ptr(), (const long long*)rng.data_ptr(), (unsigned long long*)st.data_ptr(),
                      (float*)out.data_ptr());
}
"""
CPP = "void run(torch::Tensor p, torch::Tensor rng, torch::Tensor st, torch::Tensor out, int grid, int smem);"
m = load_inline("smw_oracle", CPP, cuda_sources=SRC.replace("#include <torch/extension.h>\n", "#include <torch/types.h>\n"),
                functions=["run"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)

MB = float(sys.argv[1]) if len(sys.argv) > 1 else 28.3
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
nsm = torch.cuda.get_device_properties(0).multi_processor_count
n16 = int(MB * 1e6) // 16 // 4096 * 4096
buf = torch.randint(-2**31, 2**31 - 1, (n16 * 4,), dtype=torch.int32, device="cuda")
flush = torch.empty(200 * 2**20, dtype=torch.uint8, device="cuda")
stamps = torch.zeros(3 * nsm, dtype=torch.int64, device="cuda")
out = torch.zeros(1, device="cuda")
SMEM = 160 * 1024


def table(w):  # w: per-smid weight -> cumulative uint4 offsets, 256-uint4 (4 KB) granules
    w = torch.tensor(w, dtype=torch.float64)
    c = torch.cumsum(w / w.sum(), 0) * (n16 // 256)
    c = torch.round(c).long() * 256
    c[-1] = n16
    return torch.cat([torch.zeros(1, dtype=torch.long), c]).cuda()


def measure(rng):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(REPS)]
    loops = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for r in range(REPS + 3):
            flush.fill_(r & 255)
            if r >= 3: ev[r - 3][0].record()
            m.run(buf, rng, stamps, out, nsm, SMEM)
            if r >= 3: ev[r - 3][1].record()
            if r >= 3:
                torch.cuda.synchronize(); s = stamps.view(nsm, 3).cpu()
                loops.append(s)
        fcntl.flock(lk, fcntl.LOCK_UN)
    ts = [a.elapsed_time(b) * 1e3 for a, b in ev]
    # per-smid loop time median (us) and per-replay span (first start -> last end)
    per = {}
    spans = []
    for s in loops:
        spans.append((s[:, 2].max() - s[:, 1].min()).item() / 1e3)
        for row in s.tolist():
            per.setdefault(row[0], []).append((row[2] - row[1]) / 1e3)
    assert len(per) == nsm, f"{len(per)} distinct smids (want 1 CTA per SM)"
    lt = [st.median(per[i]) for i in range(nsm)]
    return st.median(ts), min(ts), st.median(spans), lt


eq = [1.0] * nsm
rng = table(eq)
t, tmin, span, lt = measure(rng)
print(f"MB {n16*16/1e6:.1f} nsm {nsm}: equal  event med {t:.2f} min {tmin:.2f} span {span:.2f} | loop mean {st.mean(lt):.2f} "
      f"max {max(lt):.2f} min {min(lt):.2f}")
sl = sorted(range(nsm), key=lambda i: lt[i])
print("  fastest smids", [(i, round(lt[i], 2)) for i in sl[:8]], "\n  slowest", [(i, round(lt[i], 2)) for i in sl[-8:]])
w = eq[:]
for it in range(ITERS):
    w = [w[i] / lt[i] * st.mean(lt) for i in range(nsm)]  # bytes share ~ measured rate
    t, tmin, span, lt = measure(table(w))
    print(f"weighted it{it}: event med {t:.2f} min {tmin:.2f} span {span:.2f} | loop mean {st.mean(lt):.2f} max {max(lt):.2f} "
          f"min {min(lt):.2f} | w range {min(w)/st.mean(w):.3f}-{max(w)/st.mean(w):.3f}")
t2, tmin2, span2, lt2 = measure(table(eq))
print(f"equal again: event med {t2:.2f} min {tmin2:.2f} span {span2:.2f} | loop max {max(lt2):.2f}")
torch.save({"w": w}, "/tmp/smw_w.pt")
import numpy as np
_, _, _, lt3 = measure(table(eq))
print("repeatability corr equal runs:", round(float(np.corrcoef(lt2, lt3)[0, 1]), 3), "| corr equal vs last weighted:",
      round(float(np.corrcoef(lt2, lt)[0, 1]), 3))
