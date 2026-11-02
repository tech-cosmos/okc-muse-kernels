# o45 item (1): LONG K-CHAINS. Does chaining k16 wgmmas into the SAME accumulator inside one commit group (and waiting less
# often) cut the dq || RS contention? Base = tools/s18/mb_wgmma.py (same dq, same smem B ring, 1 CTA, 2 consumer WGs).
# Variant <MODE, RT, N, CH, ORD, WD>: a group = CH k16 steps x RT 64-row tiles; dq all CH*RT fragments into A buffer,
# wgmma.fence, issue CH*RT wgmmas (ORD 0: k outer / tile inner = twm today; ORD 1: tile outer / k inner = same-acc chains
# of CH), one commit, wait_group WD (WD 1: two A buffers alternate; WD 0: one buffer). CH 1 ORD 0 WD 1 == s18/twm.
# MODE 1 dq only (xor sink), 2 MMA only, 3 both. Output: cycles per twm stage (4 k16 x RT tiles).
import torch, sys, os, subprocess, fcntl
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "s18"))
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
NS = (8, 32, 64)
# (RT, N, CH, ORD, WD)
VARS = [(2, 64, 1, 0, 1), (2, 64, 2, 1, 1), (2, 64, 4, 0, 1), (2, 64, 4, 1, 1), (2, 64, 4, 1, 0), (2, 64, 8, 0, 1),
        (2, 64, 8, 1, 1), (2, 64, 8, 1, 0), (2, 64, 16, 1, 0),
        (4, 32, 1, 0, 1), (4, 32, 2, 1, 1), (4, 32, 4, 0, 1), (4, 32, 4, 1, 1), (4, 32, 4, 1, 0), (4, 32, 8, 1, 0),
        (4, 8, 1, 0, 1), (4, 8, 4, 1, 1)]
if len(sys.argv) > 2: VARS = [v for v in VARS if f"{v[0]},{v[1]}" in sys.argv[2].split(":")]
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
template <int MODE, int RT, int N, int CH, int ORD, int WD>
__device__ __forceinline__ void group(int g, uint32_t (&A)[CH][RT][4], float (&acc)[RT][N / 2], const char* sB, const char* sW,
                                      int warp, int lane, uint32_t s0, uint32_t s1, uint32_t& xs) {
  if (MODE != 2) {
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      const int ks = g * CH + c, st = (ks >> 2) & 3;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const uint32_t q = reinterpret_cast<const uint32_t*>(sW + st * 8 * RT * 512)[((warp * RT + r) * 32 + lane) * 4 + (c & 3)] ^ g;
        dq(q, s0, s1, A[c][r]);
      }
    }
  }
  if (MODE == 1) {
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
      for (int r = 0; r < RT; ++r) xs ^= A[c][r][0] ^ A[c][r][1] ^ A[c][r][2] ^ A[c][r][3];
    return;
  }
  wg_fence();
  if (ORD == 0) {
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const int ks = g * CH + c;
        wg_rs<N>(acc[r], A[c][r], sw128_desc(sB + ((ks >> 2) & 3) * 8192) + (ks & 3) * 2);
      }
  } else {
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        const int ks = g * CH + c;
        wg_rs<N>(acc[r], A[c][r], sw128_desc(sB + ((ks >> 2) & 3) * 8192) + (ks & 3) * 2);
      }
  }
  wg_commit();
  wg_wait<WD>();
  if (MODE == 2) {
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_r(A[c][r][i]);
  }
}
template <int MODE, int RT, int N, int CH, int ORD, int WD>
__global__ void __launch_bounds__(256, 1) mbc(const uint32_t* __restrict__ w, unsigned long long* out, float* sinkf, uint32_t* sinku, int iters) {
  extern __shared__ __align__(1024) char sm[];
  char* sB = sm;
  char* sW = sm + 4 * 8192;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < (4 * 8192 + 4 * 8 * RT * 512) / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = w[i & 4095] & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  constexpr int NA = N / 2;
  float acc[RT][NA];
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t A0[CH][RT][4], A1[CH][RT][4], xs = 0;
  for (int c = 0; c < CH; ++c) for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) { A0[c][r][i] = 0x3F803F80u; A1[c][r][i] = 0x3F803F80u; }
  const uint32_t s0 = 0x3F803F80u ^ (lane & 1), s1 = 0x3F803F80u;
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    group<MODE, RT, N, CH, ORD, WD>(2 * it, A0, acc, sB, sW, warp, lane, s0, s1, xs);
    if (WD) group<MODE, RT, N, CH, ORD, WD>(2 * it + 1, A1, acc, sB, sW, warp, lane, s0, s1, xs);
    else group<MODE, RT, N, CH, ORD, WD>(2 * it + 1, A0, acc, sB, sW, warp, lane, s0, s1, xs);
  }
  wg_wait<0>();
  const unsigned long long t1 = clock64();
  float s = 0.f;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) s += acc[r][i];
  sinkf[blockIdx.x * 256 + tid] = s; sinku[blockIdx.x * 256 + tid] = xs;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in NS))
inst, go = [], []
for (rt, n, ch, o, wd) in VARS:
    for m in (1, 2, 3):
        a = f"{m}, {rt}, {n}, {ch}, {o}, {wd}"
        inst.append(f"template __global__ void mbc<{a}>(const uint32_t*, unsigned long long*, float*, uint32_t*, int);")
        go.append(f"  if (key == {m*100000 + rt*10000 + n*100 + ch} && ord == {o} && wd == {wd}) {{ auto k = mbc<{a}>; "
                  "cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); "
                  "k<<<grid, 256, SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), "
                  "(uint32_t*)su.data_ptr(), (int)iters); return 0; }")
KSRC = SRC + "\n".join(inst)
RUN = ("int64_t run(int64_t key, int64_t ord, int64_t wd, torch::Tensor w, torch::Tensor out, torch::Tensor sf, torch::Tensor su, int64_t iters, int64_t grid) {\n"
       "  const int SM = 4 * 8192 + 4 * 8 * 4 * 512 + 1024;\n" + "\n".join(go) + "\n  return -1;\n}\n")
if os.environ.get("SASS"):  # nvcc + cuobjdump: spills and HGMMA/WARPGROUP lines per kernel into /tmp/o45/
    os.makedirs("/tmp/o45", exist_ok=True); open("/tmp/o45/k.cu", "w").write(KSRC)
    r = subprocess.run("cd /tmp/o45 && /usr/local/cuda-13.0/bin/nvcc -cubin -arch=sm_90a -O3 -Xptxas -v k.cu -o k.cubin 2>&1", shell=True, capture_output=True, text=True)
    open("/tmp/o45/ptxas.txt", "w").write(r.stdout + r.stderr)
    subprocess.run("/usr/local/cuda-13.0/bin/cuobjdump -sass /tmp/o45/k.cubin > /tmp/o45/k.sass", shell=True)
    sys.exit(0)
lib = load_inline("o45_chain_" + str(abs(hash(KSRC)) % 10**8), cpp_sources="int64_t run(int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + KSRC + RUN, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
w = torch.randint(0, 2**31, (4096,), dtype=torch.int32, device="cuda")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sf = torch.zeros(132 * 256, device="cuda"); su = torch.zeros(132 * 256, dtype=torch.int32, device="cuda")
names = {1: "dq", 2: "MMA", 3: "both"}
lk = open("/tmp/okc-gpu.lock", "w")
for (rt, n, ch, o, wd) in VARS:
    IT = max(64, 4000 // ch)  # iters of 2 groups; stages = IT * 2 * CH / 4
    row = []
    for m in (1, 2, 3):
        key = m * 100000 + rt * 10000 + n * 100 + ch
        assert lib.run(key, o, wd, w, out, sf, su, 20, G) == 0
        res = []
        for rep in range(3):
            fcntl.flock(lk, fcntl.LOCK_EX)
            lib.run(key, o, wd, w, out, sf, su, IT, G); torch.cuda.synchronize()
            fcntl.flock(lk, fcntl.LOCK_UN)
            res.append(out[:G].float().mean().item() / (IT * 2 * ch / 4))
        row.append(f"{names[m]} {sorted(res)[1]:6.0f}")
    print(f"RT{rt} N{n:2d} CH{ch:2d} {'chain' if o else 'tiles'} WD{wd}: " + " | ".join(row), flush=True)
