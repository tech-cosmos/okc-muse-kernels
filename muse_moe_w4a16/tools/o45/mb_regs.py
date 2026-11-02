# o45 item (2): PHYSICAL REGISTER PLACEMENT. s18 twm-style stage (LDS.128 per stage, per k16 step: dq RT fragments,
# fence, RT RS wgmmas, commit, wait 1; two A sets) with variants that keep the instruction multiset but move ptxas'
# register allocation: PAD = live dummy registers (no instructions: empty asm "+r" each step) declared before the
# accumulators, PERM bit0 = swap A-set roles, bit1 = reversed tile->accumulator map, bit2 = dq temps computed in reverse
# order (o3..o0) and bmul order reversed. MODE 1 dq only (xor sink), 2 MMA only, 3 both. cycles per stage (4 k16 x RT).
import torch, sys, os, subprocess, fcntl, re, collections, hashlib
from torch.utils.cpp_extension import load_inline
def gen_rs(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
SHAPES = [(2, 64), (4, 32)]
VARS = [(p, q) for p in (0, 1, 3, 6, 11) for q in (0, 2, 4, 6)][:18]  # (PAD, PERM); PERM bit0 (A-set swap) = identical SASS
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
template <bool REV> __device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  if (!REV) {
    const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
    const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
    a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
  } else {
    const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
    const uint32_t o2 = __funnelshift_l(q, q, 13) & M, o1 = __funnelshift_l(q, q, 10) & M, o0 = q & M;
    a[3] = bmul(o3, s1); a[2] = bmul(o2, s0); a[1] = bmul(o1, s1); a[0] = bmul(o0, s0);
  }
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
template <int MODE, int RT, int N, int PAD, int PERM>
__global__ void __launch_bounds__(256, 1) mbr(const uint32_t* __restrict__ w, unsigned long long* out, float* sinkf, uint32_t* sinku, int iters) {
  extern __shared__ __align__(1024) char sm[];
  char* sB = sm;
  char* sW = sm + 4 * 8192;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < (4 * 8192 + 4 * 8 * RT * 512) / 4; i += 256) reinterpret_cast<uint32_t*>(sm)[i] = w[i & 4095] & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  constexpr int NA = N / 2;
  constexpr bool SWAP = PERM & 1, RTREV = PERM & 2, DQREV = PERM & 4;
  uint32_t pad[PAD > 0 ? PAD : 1];
#pragma unroll
  for (int i = 0; i < PAD; ++i) pad[i] = w[(tid * 7 + i) & 4095];
  float acc[RT][NA];
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[RT][4], afB[RT][4], xs = 0;
  for (int r = 0; r < RT; ++r) for (int i = 0; i < 4; ++i) { afA[r][i] = 0x3F803F80u; afB[r][i] = 0x3F803F80u; }
  const uint32_t s0 = 0x3F803F80u ^ (lane & 1), s1 = 0x3F803F80u;
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
      const bool odd = (j & 1) ^ SWAP;
#pragma unroll
      for (int i = 0; i < PAD; ++i) fence_r(pad[i]);
      if (MODE != 2) {
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const uint32_t q = (j == 0 ? wv[r].x : j == 1 ? wv[r].y : j == 2 ? wv[r].z : wv[r].w) ^ it;
          if (odd) dq<DQREV>(q, s0, s1, afB[r]); else dq<DQREV>(q, s0, s1, afA[r]);
        }
      }
      if (MODE == 1) {
#pragma unroll
        for (int r = 0; r < RT; ++r) xs ^= (odd ? afB[r][0] ^ afB[r][1] ^ afB[r][2] ^ afB[r][3] : afA[r][0] ^ afA[r][1] ^ afA[r][2] ^ afA[r][3]);
      } else {
        wg_fence();
#pragma unroll
        for (int r = 0; r < RT; ++r) wg_rs<N>(acc[RTREV ? RT - 1 - r : r], odd ? afB[r] : afA[r], d0 + j * 2);
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
  for (int r = 0; r < RT; ++r) for (int i = 0; i < NA; ++i) s += acc[r][i];
#pragma unroll
  for (int i = 0; i < PAD; ++i) xs ^= pad[i];
  sinkf[blockIdx.x * 256 + tid] = s; sinku[blockIdx.x * 256 + tid] = xs;
  if (tid == 0) out[blockIdx.x] = t1 - t0;
}
"""
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n) for n in (32, 64)))
inst, go = [], []
for (rt, n) in SHAPES:
    for (p, q) in VARS:
        for m in (1, 2, 3):
            a = f"{m}, {rt}, {n}, {p}, {q}"
            inst.append(f"template __global__ void mbr<{a}>(const uint32_t*, unsigned long long*, float*, uint32_t*, int);")
            go.append(f"  if (m == {m} && rt == {rt} && n == {n} && p == {p} && q == {q}) {{ auto k = mbr<{a}>; "
                      "cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); "
                      "k<<<grid, 256, SM>>>((const uint32_t*)w.data_ptr(), (unsigned long long*)out.data_ptr(), (float*)sf.data_ptr(), "
                      "(uint32_t*)su.data_ptr(), (int)iters); return 0; }")
KSRC = SRC + "\n".join(inst)
RUN = ("int64_t run(int64_t m, int64_t rt, int64_t n, int64_t p, int64_t q, torch::Tensor w, torch::Tensor out, torch::Tensor sf, torch::Tensor su, int64_t iters, int64_t grid) {\n"
       "  const int SM = 4 * 8192 + 4 * 8 * 4 * 512 + 1024;\n" + "\n".join(go) + "\n  return -1;\n}\n")
# SASS: per kernel, register count, instruction multiset hash, register-mapping hash (dq ALU + HGMMA operands)
os.makedirs("/tmp/o45r", exist_ok=True); open("/tmp/o45r/k.cu", "w").write(KSRC)
subprocess.run("cd /tmp/o45r && /usr/local/cuda-13.0/bin/nvcc -cubin -arch=sm_90a -O3 -Xptxas -v k.cu -o k.cubin > ptxas.txt 2>&1 && "
               "/usr/local/cuda-13.0/bin/cuobjdump -sass k.cubin > k.sass", shell=True, check=True)
sass, fn = collections.defaultdict(list), None
for l in open("/tmp/o45r/k.sass"):
    mm = re.search(r"Function : \S*mbrILi(\d)ELi(\d)ELi(\d+)ELi(\d+)ELi(\d)E", l)
    if mm: fn = tuple(int(x) for x in mm.groups()); continue
    if fn and re.search(r"/\*[0-9a-f]{4,}\*/", l): sass[fn].append(l.split("*/", 1)[1].split(";")[0].strip())
regs = {}
cur = None
for l in open("/tmp/o45r/ptxas.txt"):
    mm = re.search(r"Compiling entry function '\S*mbrILi(\d)ELi(\d)ELi(\d+)ELi(\d+)ELi(\d)E", l)
    if mm: cur = tuple(int(x) for x in mm.groups())
    mm = re.search(r"Used (\d+) registers", l)
    if mm and cur: regs[cur] = int(mm.group(1))
    if "spill" in l and cur and not " 0 bytes spill stores, 0 bytes spill loads" in l: print("SPILL", cur, l.strip())
def info(k):
    ins = sass[k]
    ops = collections.Counter(i.split()[0] for i in ins if i)
    body = [i for i in ins if re.match(r"(LOP3|SHF|IMAD|HMUL2|HGMMA|LDS)", i)]
    mh = hashlib.md5("\n".join(body).encode()).hexdigest()[:6]
    dep = sum("DEPBAR.LE gsb0, 0x0" in i for i in ins)
    return regs.get(k), sum(ops.values()), sum(v for o, v in ops.items() if o.startswith("HGMMA")), sum(v for o, v in ops.items() if o.startswith(("LOP3", "HMUL2", "SHF", "IMAD"))), mh, dep
if os.environ.get("SASSONLY"):
    for (rt, n) in SHAPES:
        for (p, q) in VARS: print(rt, n, p, q, [info((m, rt, n, p, q)) for m in (1, 3)])
    sys.exit(0)
lib = load_inline("o45_regs_" + hashlib.md5(KSRC.encode()).hexdigest()[:8], cpp_sources="int64_t run(int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + KSRC + RUN, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
w = torch.randint(0, 2**31, (4096,), dtype=torch.int32, device="cuda")
G = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = torch.zeros(132, dtype=torch.int64, device="cuda"); sf = torch.zeros(132 * 256, device="cuda"); su = torch.zeros(132 * 256, dtype=torch.int32, device="cuda")
lk = open("/tmp/okc-gpu.lock", "w"); IT = 2000
for (rt, n) in SHAPES:
    for (p, q) in VARS:
        row = []
        for m in (1, 2, 3):
            assert lib.run(m, rt, n, p, q, w, out, sf, su, 20, G) == 0
            res = []
            for rep in range(3):
                fcntl.flock(lk, fcntl.LOCK_EX); lib.run(m, rt, n, p, q, w, out, sf, su, IT, G); torch.cuda.synchronize(); fcntl.flock(lk, fcntl.LOCK_UN)
                res.append(out[:G].float().mean().item() / IT)
            row.append(f"{sorted(res)[1]:6.0f}")
        i1, i3 = info((1, rt, n, p, q)), info((3, rt, n, p, q))
        print(f"RT{rt} N{n} PAD{p} PERM{q}: dq {row[0]} MMA {row[1]} both {row[2]} | regs {i3[0]} instr {i3[1]} hg {i3[2]} lop+hmul {i3[3]} map {i3[4]} dq-map {i1[4]} dep0 {i3[5]}", flush=True)
