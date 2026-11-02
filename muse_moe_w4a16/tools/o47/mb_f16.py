# o47 (lead 11-02 09:10Z) MEASUREMENT ONLY: does dq overlap better with the tensor cores when the MMA is FP16?
# Base = o45/mb_regs.py PAD0 PERM0 (= s18 twm stage: LDS.128 per stage, per k16 step dq RT fragments, fence, RT RS wgmmas,
# commit, wait 1, two A sets). TY 0: bf16 dq (mul.rn.bf16x2) + wgmma f32.bf16.bf16 (today)
# TY 1: f16 dq (same LOP3/SHF/IMAD multiset, mask for the f16 exponent field, mul.rn.f16x2) + wgmma f32.f16.f16
# TY 2: f16 dq + wgmma f16.f16.f16 (D = N/4 regs). MODE 1 dq only (xor sink), 2 MMA only, 3 both. Cycles per stage (4 k16 x RT).
import torch, sys, os, subprocess, fcntl, re, collections, hashlib
from torch.utils.cpp_extension import load_inline
def gen_rs(n, ty):
    f16acc = ty == 2
    na = n // 4 if f16acc else n // 2
    dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join((f'"+r"(d[{i}])' if f16acc else f'"+f"(d[{i}])') for i in range(na))
    dt = "f16.f16.f16" if ty == 2 else ("f32.f16.f16" if ty == 1 else "f32.bf16.bf16")
    at = "uint32_t" if f16acc else "float"
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}, {ty}>({at}* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.{dt} {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
SHAPES = [(4, 8), (4, 32), (2, 64)]
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
template <int TY> __device__ __forceinline__ uint32_t hmul(uint32_t a, uint32_t b) {
  uint32_t r;
  if (TY == 0) asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  else asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r; }
// bf16: code bits at sign + bits 8..6 (value = E2M1 * 2^-126); f16: sign + bits 11..9 (value = E2M1 * 2^-14, exact incl.
// the 0.5 subnormal). Same instruction multiset; prepare() would repack the nibbles so these shifts land them.
template <int TY> __device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = TY == 0 ? 0x81C081C0u : 0x8E008E00u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, TY == 0 ? 10 : 7) & M, o2 = __funnelshift_l(q, q, TY == 0 ? 13 : 10) & M;
  const uint32_t o3 = ((q & (TY == 0 ? 0x001B001Bu : 0x00DB00DBu)) * (TY == 0 ? 16648u : 2081u)) & M;
  a[0] = hmul<TY>(o0, s0); a[1] = hmul<TY>(o1, s1); a[2] = hmul<TY>(o2, s0); a[3] = hmul<TY>(o3, s1);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N, int TY> __device__ __forceinline__ void wg_rs(typename std::conditional<TY == 2, uint32_t, float>::type* d, const uint32_t* a, uint64_t desc);
@@GEN@@
template <int MODE, int RT, int N, int TY>
__global__ void __launch_bounds__(256, 1) mbf(const uint32_t* __restrict__ w, unsigned long long* out, float* sinkf, uint32_t* sinku, int iters) {
  extern __shared__ __align__(1024) char sm[];
  char* sB = sm;
  char* sW = sm + 4 * 8192;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < (4 * 8192 + 4 * 8 * RT * 512) / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = w[i & 4095] & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  using AT = typename std::conditional<TY == 2, uint32_t, float>::type;
  constexpr int NA = TY == 2 ? N / 4 : N / 2;
  AT acc[RT][NA];
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) acc[r][i] = 0;
  uint32_t afA[RT][4], afB[RT][4], xs = 0;
  const uint32_t one = TY == 0 ? 0x3F803F80u : 0x3C003C00u;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) { afA[r][i] = one; afB[r][i] = one; }
  const uint32_t s0 = one ^ (lane & 1), s1 = one;
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    const int st = it & 3;
    const char* sb = sB + st * 8192;
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sW + st * 8 * RT * 512)[(warp * RT + r) * 32 + lane];
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool odd = j & 1;
      if (MODE != 2) {
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) ^ it;
          if (odd) dq<TY == 0 ? 0 : 1>(q, s0, s1, afB[r]); else dq<TY == 0 ? 0 : 1>(q, s0, s1, afA[r]);
        }
      }
      if (MODE == 1) {
#pragma unroll
        for (int r = 0; r < RT; ++r) xs ^= (odd ? afB[r][0] ^ afB[r][1] ^ afB[r][2] ^ afB[r][3] : afA[r][0] ^ afA[r][1] ^ afA[r][2] ^ afA[r][3]);
      } else {
        wg_fence();
#pragma unroll
        for (int r = 0; r < RT; ++r) wg_rs<N, TY>(acc[r], odd ? afB[r] : afA[r], d0 + j * 2);
        wg_commit();
        wg_wait<1>();
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) { if (odd) fence_r(afA[r][i]); else fence_r(afB[r][i]); }
      }
    }
  }
  wg_wait<0>();
  const unsigned long long t1 = clock64();
  float s = 0.f;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) { if (TY == 2) s += __uint_as_float(acc[r][i]); else s += acc[r][i]; }
  sinkf[blockIdx.x * 256 + tid] = s; sinku[blockIdx.x * 256 + tid] = xs;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
}
"""
SRC = "#include <type_traits>\n" + SRC.replace("@@GEN@@", "".join(gen_rs(n, ty) for n in (8, 32, 64) for ty in (0, 1, 2)))
inst, go = [], []
for (rt, n) in SHAPES:
    for ty in (0, 1, 2):
        for m in (1, 2, 3):
            a = f"{m}, {rt}, {n}, {ty}"
            inst.append(f"template __global__ void mbf<{a}>(const uint32_t*, unsigned long long*, float*, uint32_t*, int);")
            go.append(f"  if (m == {m} && rt == {rt} && n == {n} && ty == {ty}) {{ auto k = mbf<{a}>; "
                      "cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); "
                      "k<<<grid, 256, SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), "
                      "(uint32_t*)su.data_ptr(), (int)iters); return 0; }")
KSRC = SRC + "\n".join(inst)
RUN = ("int64_t run(int64_t m, int64_t rt, int64_t n, int64_t ty, torch::Tensor w, torch::Tensor out, torch::Tensor sf, torch::Tensor su, int64_t iters, int64_t grid) {\n"
       "  const int SM = 4 * 8192 + 4 * 8 * 4 * 512 + 1024;\n" + "\n".join(go) + "\n  return -1;\n}\n")
os.makedirs("/tmp/o47", exist_ok=True); open("/tmp/o47/k.cu", "w").write(KSRC)
subprocess.run("cd /tmp/o47 && /usr/local/cuda-13.0/bin/nvcc -cubin -arch=sm_90a -O3 -Xptxas -v k.cu -o k.cubin > ptxas.txt 2>&1 && "
               "/usr/local/cuda-13.0/bin/cuobjdump -sass k.cubin > k.sass", shell=True, check=True)
sass, fn = collections.defaultdict(list), None
pat = r"mbfILi(\d)ELi(\d)ELi(\d+)ELi(\d)E"
for l in open("/tmp/o47/k.sass"):
    mm = re.search(r"Function : \S*" + pat, l)
    if mm: fn = tuple(int(x) for x in mm.groups()); continue
    if fn and re.search(r"/\*[0-9a-f]{4,}\*/", l): sass[fn].append(l.split("*/", 1)[1].split(";")[0].strip())
regs, cur = {}, None
for l in open("/tmp/o47/ptxas.txt"):
    mm = re.search(r"Compiling entry function '\S*" + pat, l)
    if mm: cur = tuple(int(x) for x in mm.groups())
    mm = re.search(r"Used (\d+) registers", l)
    if mm and cur: regs[cur] = int(mm.group(1))
    if "spill" in l and cur and " 0 bytes spill stores, 0 bytes spill loads" not in l: print("SPILL", cur, l.strip())
def info(k):
    ins = sass[k]
    ops = collections.Counter(i.split()[0] for i in ins if i)
    hg = sorted(set(i.split()[0] for i in ins if i.startswith("HGMMA")))
    hm = sorted(set(i.split()[0] for i in ins if i.startswith("HMUL2")))
    nalu = sum(v for o, v in ops.items() if o.startswith(("LOP3", "HMUL2", "SHF", "IMAD")))
    return regs.get(k), sum(ops.values()), nalu, hg, hm
if os.environ.get("SASSONLY"):
    for (rt, n) in SHAPES:
        for ty in (0, 1, 2): print(rt, n, ty, info((1, rt, n, ty)), info((3, rt, n, ty)))
    sys.exit(0)
lib = load_inline("o47_f16_" + hashlib.md5(KSRC.encode()).hexdigest()[:8], cpp_sources="int64_t run(int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + KSRC + RUN, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
w = torch.randint(0, 2**31, (4096,), dtype=torch.int32, device="cuda")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 132
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sf = torch.zeros(132 * 256, device="cuda"); su = torch.zeros(132 * 256, dtype=torch.int32, device="cuda")
lk = open("/tmp/okc-gpu.lock", "w"); IT = 2000
TYN = {0: "bf16/f32acc", 1: "f16/f32acc ", 2: "f16/f16acc "}
for (rt, n) in SHAPES:
    base = None
    for ty in (0, 1, 2):
        row = []
        for m in (1, 2, 3):
            assert lib.run(m, rt, n, ty, w, out, sf, su, 20, G) == 0
            res = []
            for rep in range(5):
                fcntl.flock(lk, fcntl.LOCK_EX); lib.run(m, rt, n, ty, w, out, sf, su, IT, G); torch.cuda.synchronize(); fcntl.flock(lk, fcntl.LOCK_UN)
                res.append(out[:G].float().mean().item() / IT)
            row.append(sorted(res)[2])
        if ty == 0: base = row[2]
        i3 = info((3, rt, n, ty))
        print(f"RT{rt} N{n:2d} {TYN[ty]}: dq {row[0]:6.0f} MMA {row[1]:6.0f} both {row[2]:6.0f} both/today {row[2] / base:.3f} | regs {i3[0]} alu {i3[2]} {i3[3]} {i3[4]}", flush=True)
