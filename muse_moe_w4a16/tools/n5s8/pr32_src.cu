#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef G30_NS
#define G30_NS 10
#endif
#ifndef G30_NW
#define G30_NW 10
#endif
#ifndef G30_PRE
#define G30_PRE 1  // proj s94: consumers read + sort the escape header before griddepcontrol.wait
#endif
#ifndef G30_ESC
#define G30_ESC 8
#endif
#ifndef G30_MTOK
#define G30_MTOK 128
#endif
#define G30_ND (G30_MTOK / 2)
#ifndef G30_S
#define G30_S 1  // proj s95: K split over a (1, S, 1) cluster, rank 1 sums into rank 0 via DSMEM (S <= 2)
#endif
#define G30_XBOX (G30_MTOK * 128)
#define G30_WST 6144

namespace pr32 {
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile(
      "{\n .reg .pred p;\n PR32_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra PR32_WAIT;\n}\n" :: "r"(smem_u32(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(smem_u32(dst)), "l"(m), "r"(c0), "r"(c1), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bulk1d(void* dst, const void* src, uint32_t n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(n), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ uint64_t dsw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
// patch the half (kh, rr, h) of j's A fragment with BF16 bits vb: a[4 j + 2 kh + rr] (m16n8k16 A layout), h = high half
__device__ __forceinline__ void esc_patch(unsigned (&a)[16], unsigned key) {
  const int kk = (key >> 18) & 63, idx = 4 * (kk >> 4) + 2 * ((kk >> 3) & 1) + ((key >> 16) & 1);
  const unsigned vb = key & 0xFFFFu, sel = (kk & 1) ? 0x5410u : 0x3254u;
#pragma unroll
  for (int u = 0; u < 16; u++)
    if (u == idx) a[u] = __byte_perm(a[u], vb, sel);
}
template <int P>
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel3) {
  const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
  const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);
  unsigned p;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
  return lop_aob(sh, 0x87008700u, p);
}
#define G30_D8(i) "+f"(d[i]), "+f"(d[i + 1]), "+f"(d[i + 2]), "+f"(d[i + 3]), "+f"(d[i + 4]), "+f"(d[i + 5]), "+f"(d[i + 6]), "+f"(d[i + 7])
__device__ __forceinline__ void wg_mma_rs(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint64_t db) {
#if G30_MTOK == 128
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
      : G30_D8(0), G30_D8(8), G30_D8(16), G30_D8(24), G30_D8(32), G30_D8(40), G30_D8(48), G30_D8(56)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#elif G30_MTOK == 32
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
      " {%16,%17,%18,%19}, %20, 1, 1, 1, 0;\n"
      : G30_D8(0), G30_D8(8)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#elif G30_MTOK == 16
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7},"
      " {%8,%9,%10,%11}, %12, 1, 1, 1, 0;\n"
      : G30_D8(0)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#elif G30_MTOK == 8
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3},"
      " {%4,%5,%6,%7}, %8, 1, 1, 1, 0;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#else
#error G30_MTOK
#endif
}

#define PR_RST 8192                 // router stage: 64 rows x 64 k raw BF16 in A-fragment order [4 j][128 thr][16 B]
#define PR_SLOT (PR_RST > G30_WST ? PR_RST : G30_WST)
#ifndef PR_F32
#define PR_F32 0  // n5s8 (T = 8): FP32 router (FP32 input xr, FP32 rows, FMA), SGLang's T <= 8 precision
#endif
#if PR_F32 && G30_MTOK != 8
#error PR_F32 needs G30_MTOK 8
#endif
#define PR_NG (PR_F32 ? 4 : 8)                // router 64-row groups (F32: 256 rows; else hi 256 | lo 256)
#define PR_GST (PR_F32 ? 256 : 128)           // stages per group (F32: 32 k x 64 rows FP32; else 64 k x 64 rows BF16)
#define PR_RKS (64 * G30_S / PR_NG)  // router K slices per 64-row group (NG groups x RKS = 64 S CTAs)
#ifndef PR_ROUTE
#define PR_ROUTE 1
#endif
#ifndef PR_TICKET
#define PR_TICKET 1
#endif
#ifndef PR_ROUTER
#define PR_ROUTER 1
#endif
#ifndef PR_SA
#define PR_SA 1  // n5s3: pair handoffs by st.async + mbarrier complete_tx (rank 1 fires and exits; no fence/bar.sync)
#endif
#ifndef PR_HL
#define PR_HL 0  // n5s4: router rows packed as [hi 32 | lo 32] per 64-row group: hi + lo folded in-CTA, half the partial bytes
#endif
#define PR_RW (PR_HL ? 32 : 64)  // pair partial width (logits per 64-row group)
#define PR_RBS (PR_SA ? 3 : G30_S - 1)  // pooler partial slots in rbuf (SA: rank 1 wg0, rank 1 wg1, rank 0 wg1)
#define PR_NR (PR_ROUTER ? PR_GST / PR_RKS : 0)  // router stages per CTA
#ifndef PR_IL
#define PR_IL PR_F32  // n5s8: F32 router stages interleaved with the first pooler stages (P R P R per WG: FMA under wgmma)
#endif
// stage i -> router? and its ordinal among its kind
#if PR_IL
#define PR_ISR(i) ((i) < 2 * PR_NR && (((i) >> 1) & 1))
#define PR_ORD(i) ((i) < 2 * PR_NR ? (((i) >> 2) * 2 + ((i) & 1)) : (i) - PR_NR)
#else
#define PR_ISR(i) ((i) < PR_NR)
#define PR_ORD(i) ((i) < PR_NR ? (i) : (i) - PR_NR)
#endif
constexpr int E = 256, TOPK = 8;

struct RArgs {
  const uint8_t* rw; float* part; float* logits; int* cnt; const float* bias; const int* ls;
  int* ids; int* lids; float* wts; float mult, reps; int local; unsigned long long* tl;
};
__device__ __forceinline__ void st_async4(uint32_t a, float x, float y, float z, float w, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(a), "f"(x), "f"(y), "f"(z), "f"(w), "r"(mb) : "memory");
}
__device__ __forceinline__ float4 lds4f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ unsigned long long gtime() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#ifdef PR_TL
#define TLX(a_, k) a_.tl[(blockIdx.x * G30_S + blockIdx.y) * 12 + (k)] = gtime();
#define TLM(k) if (t == 0) TLX(ra, k)
#else
#define TLX(a_, k) {}
#define TLM(k) {}
#endif

// one token per warp: scores = sigmoid(MULT logit), top-8 of score + bias (lowest index on ties: same picks as the
// layer's _routing), weights = picked / (sum + REPS) (sum in pick order). Keys as order-preserving u32 (+0.0f folds -0
// into +0, like the float compare): one redux max + one redux min (index among the tied lanes) per pick.
__device__ __forceinline__ void route_token(const RArgs& a, const float* lg, const float* bs_, int tok, int lane, int ls) {
  float sc[E / 32];
  unsigned key[E / 32];
#pragma unroll
  for (int i = 0; i < E / 32; ++i) {
    const int e = lane + 32 * i;
#if PR_HL || PR_F32
    const float l = lg[e];  // hi + lo already folded (F32: one logit)
#else
    const float l = lg[e] + lg[E + e];  // hi + lo
#endif
    sc[i] = 1.f / (1.f + expf(-l * a.mult));
    const unsigned u = __float_as_uint(sc[i] + bs_[e] + 0.0f);
    key[i] = u ^ ((unsigned)((int)u >> 31) | 0x80000000u);
  }
  int myid = 0; float mysc = 0.f, sum = 0.f;
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    unsigned bk = key[0]; int bi = 0;
#pragma unroll
    for (int i = 1; i < E / 32; ++i)
      if (key[i] > bk) { bk = key[i]; bi = i; }
    const unsigned m = __reduce_max_sync(0xffffffffu, bk);
    const unsigned w = __reduce_min_sync(0xffffffffu, bk == m ? (unsigned)(lane + 32 * bi) : 0xffffffffu);
    float bs = 0.f;
#pragma unroll
    for (int i = 0; i < E / 32; ++i) if (i == bi) bs = sc[i];
    bs = __shfl_sync(0xffffffffu, bs, w & 31);
    if (lane == (int)(w & 31)) {
#pragma unroll
      for (int i = 0; i < E / 32; ++i) if (i == (int)(w >> 5)) key[i] = 0u;
    }
    if (lane == k) { myid = (int)w; mysc = bs; }
    sum += bs;
  }
  if (lane < TOPK) {
    a.ids[tok * TOPK + lane] = myid;
    const int loc = (myid - ls + E) % E;
    a.lids[tok * TOPK + lane] = loc < a.local ? loc : -1;
    a.wts[tok * TOPK + lane] = mysc / (sum + a.reps);
  }
}

// router partial element idx of [T][PR_RW]: WG0 + WG1 accumulators ([2][T][64]); PR_HL: + the lo row (r + 32) too
__device__ __forceinline__ float4 rsum4(const float* rred, int idx) {
#if PR_HL
  const int b = (idx >> 5) * 64 + (idx & 31);
  const float4 a = *(const float4*)(rred + b), c = *(const float4*)(rred + G30_MTOK * 64 + b);
  const float4 e = *(const float4*)(rred + b + 32), h = *(const float4*)(rred + G30_MTOK * 64 + b + 32);
  return make_float4((a.x + c.x) + (e.x + h.x), (a.y + c.y) + (e.y + h.y), (a.z + c.z) + (e.z + h.z), (a.w + c.w) + (e.w + h.w));
#else
  const float4 u = *(const float4*)(rred + idx), v = *(const float4*)(rred + G30_MTOK * 64 + idx);
  return make_float4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
#endif
}

// finishers (clusters x = 0, 1: 4 CTAs on the 4 SMs the 128 streaming CTAs leave free): f = 2 x + y routes tokens
// 8 f .. 8 f + 7. Waits for the 64 pair partials, pulls its 8-token chunks of all of them (64 x 2 KB) in one bulk shot,
// sums them in pair-slice order (deterministic), routes one token per warp.
__device__ __forceinline__ void finisher(const RArgs& a, uint8_t* sm, uint64_t* fb, float* bs_, float* lgb) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, f = 2 * blockIdx.x + blockIdx.y;
  constexpr int TPF = G30_MTOK / 4, CH = TPF * PR_RW * 4, NP = PR_RKS / 2;  // tokens per finisher, chunk bytes, pair slices per rg
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid < E) bs_[tid] = __ldg(a.bias + tid);
  const int ls = __ldg(a.ls);
  if (tid == 0) {
    int v;
    do {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(a.cnt) : "memory");
      if (v < 64) __nanosleep(32);
    } while (v < 64);
    TLX(a, 6)
    int o;
    asm volatile("atom.add.gpu.global.s32 %0, [%1], 1;" : "=r"(o) : "l"(a.cnt + 1) : "memory");
    if (o == 3) { a.cnt[0] = 0; a.cnt[1] = 0; }  // all 4 finishers are past their poll
    asm volatile("fence.proxy.async.global;" ::: "memory");
    mbar_expect_tx(fb, 64 * CH);
    for (int i = 0; i < 64; i++)
      bulk1d(sm + i * CH, (const uint8_t*)a.part + ((size_t)i * G30_MTOK + TPF * f) * PR_RW * 4, CH, fb);
  }
  __syncthreads();
  mbar_wait(fb, 0);
  if (tid == 0) TLX(a, 2)
  if (warp < TPF) {
    const float* c = (const float*)sm;
    float* lg = lgb + warp * 2 * E;  // this warp's token: [512] logits (PR_HL: [256] full logits)
    constexpr int NL = (PR_HL || PR_F32) ? E : 2 * E;
#pragma unroll
    for (int i = 0; i < NL / 32; i++) {
      const int row = lane + 32 * i, rg = row / PR_RW, r = row % PR_RW;
      float s = 0.f;
#pragma unroll
      for (int p = 0; p < NP; p++) s += c[((rg * NP + p) * TPF + warp) * PR_RW + r];
      lg[row] = s;
    }
    __syncwarp();
    if (tid == 0) TLX(a, 3)
    route_token(a, lg, bs_, TPF * f + warp, lane, ls);
#pragma unroll
    for (int i = 0; i < NL / 32; i++) __stcg(a.logits + (TPF * f + warp) * 2 * E + lane + 32 * i, lg[lane + 32 * i]);
    if (tid == 0) TLX(a, 7)
  }
}

// grid (64, S) in (1, S, 1) clusters: CTA (g, y) = pooler rows 64 g.., K part y (S = 2: halves, DSMEM sum into rank 0)
// FIRST does router item r = S g + y: router rows (hi 0..255 | lo 256..511) 64 (r / RKS).., K slice r % RKS (NR stages).
__global__ void __launch_bounds__(352, 1)
pr32_kernel(const __grid_constant__ CUtensorMap tmX, const __grid_constant__ CUtensorMap tmXR, const uint8_t* __restrict__ wp, __nv_bfloat16* __restrict__ out,
            const float2* __restrict__ hdr, int N, int KS, float scale, const RArgs ra) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm;                                  // NS x XBOX
  uint8_t* ws = sm + G30_NS * G30_XBOX;              // NW x SLOT
  uint64_t* full = (uint64_t*)(ws + G30_NW * PR_SLOT);
  uint64_t* empty = full + G30_NS;
  uint64_t* wfull = empty + G30_NS;
  uint64_t* wempty = wfull + G30_NW;
  uint64_t* rbar = wempty + G30_NW;
  uint64_t* pbar = rbar + 1;  // router pair partial from rank 1
  uint64_t* fbar = rbar + 2;  // finisher bulk ingest
  float* rred = (float*)(ws + G30_NW * PR_SLOT + 1024);            // [2 WG][T][64] router partials
  float* rpair = rred + 2 * G30_MTOK * 64;   // [T][64] rank 1 router partial (DSMEM)
  float* rbuf = rpair + G30_MTOK * 64;                               // [S - 1][ND][128] pooler partials (DSMEM)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = (int)blockIdx.x - 2;  // x = 0, 1: the finishers
  const int i0 = 2 * ((int)blockIdx.y * (KS / 2) / G30_S), i1 = 2 * (((int)blockIdx.y + 1) * (KS / 2) / G30_S);
  const int r = G30_S * g + (int)blockIdx.y, rg = r / PR_RKS, rks = r % PR_RKS;
#ifdef PR_NOPOOL
  const int nst = PR_NR;
#else
  const int nst = PR_NR + (i1 - i0);
#endif
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid == 0) TLX(ra, 8)
  if (tid == 0) {
    for (int i = 0; i < G30_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    for (int i = 0; i < G30_NW; i++) { mbar_init(&wfull[i], 1); mbar_init(&wempty[i], 1); }
    mbar_init(rbar, G30_S > 1 ? G30_S - 1 : 1);
    mbar_init(pbar, 1);
    mbar_init(fbar, 1);
#if PR_SA
    if (blockIdx.y == 0 && blockIdx.x >= 2) {  // bytes the pushes will complete (may land before or after this)
      mbar_expect_tx(rbar, 3 * G30_ND * 128 * 4);
      mbar_expect_tx(pbar, G30_MTOK * PR_RW * 4);
    }
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#if G30_S > 1
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#else
  __syncthreads();
#endif
  if (g < 0) { finisher(ra, sm, fbar, rbuf + G30_ND * 128 * PR_RBS, rred); return; }
  if (warp == 10) {  // router partial: (WG0 + WG1) [+ rank 1's via DSMEM] -> part[rg][pair slice], then count it
    asm volatile("bar.sync 3, 288;" ::: "memory");
    if (blockIdx.y == 1) {
      uint32_t rb, rm;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rb) : "r"(smem_u32(rpair)));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rm) : "r"(smem_u32(pbar)));
      for (int idx = lane * 4; idx < G30_MTOK * PR_RW; idx += 128) {
        const float4 v = rsum4(rred, idx);
#if PR_SA
        st_async4(rb + idx * 4, v.x, v.y, v.z, v.w, rm);
#else
        asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(rb + idx * 4), "f"(v.x), "f"(v.y),
                     "f"(v.z), "f"(v.w) : "memory");
#endif
      }
#if !PR_SA
      __syncwarp();
      if (lane == 0) asm volatile("fence.acq_rel.cluster;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(rm) : "memory");
#endif
      return;
    }
    asm volatile(
        "{\n .reg .pred p;\n PR32_PW:\n"
        " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra PR32_PW;\n}\n" :: "r"(smem_u32(pbar)) : "memory");
    float* pp = ra.part + (size_t)(rg * (PR_RKS / 2) + (g % (PR_RKS / 2))) * G30_MTOK * PR_RW;
    if (lane == 0) TLX(ra, 0)
    for (int idx = lane * 4; idx < G30_MTOK * PR_RW; idx += 128) {
      const float4 v = rsum4(rred, idx);
      const float4 w = *(const float4*)(rpair + idx);
      __stcg((float4*)(pp + idx), make_float4(v.x + w.x, v.y + w.y, v.z + w.z, v.w + w.w));
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(ra.cnt) : "memory");
    if (lane == 0) TLX(ra, 1)
    return;
  }
  if (warp == 8) {  // W: router stages then pooler stages (weights only: before griddepcontrol.wait)
    if (lane == 0) {
#ifdef PR_L2W  // H200 proxy: every CTA streams group (g & 3)'s weights (~3 MB, L2-resident) = consumer-bound floor
      const uint8_t* rsrc = ra.rw + ((size_t)(rg & 0) * PR_GST + (size_t)rks * PR_NR) * PR_RST;
      const uint8_t* src = wp + ((size_t)(g & 3) * KS + i0) * G30_WST;
#else
      const uint8_t* rsrc = ra.rw + ((size_t)rg * PR_GST + (size_t)rks * PR_NR) * PR_RST;
      const uint8_t* src = wp + ((size_t)g * KS + i0) * G30_WST;
#endif
      for (int i = 0; i < nst; i++) {
        const int st = i % G30_NW, ph = (i / G30_NW) & 1;
        mbar_wait(&wempty[st], ph ^ 1);
        if (PR_ISR(i)) {
          mbar_expect_tx(&wfull[st], PR_RST);
          bulk1d(ws + st * PR_SLOT, rsrc + (size_t)PR_ORD(i) * PR_RST, PR_RST, &wfull[st]);
        } else {
          mbar_expect_tx(&wfull[st], G30_WST);
          bulk1d(ws + st * PR_SLOT, src + (size_t)PR_ORD(i) * G30_WST, G30_WST, &wfull[st]);
        }
      }
    }
    return;
  }
  if (warp == 9) {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // x belongs to the previous kernels
    if (lane == 0) {
      for (int i = 0; i < nst; i++) {
        const int st = i % G30_NS, ph = (i / G30_NS) & 1;
        mbar_wait(&empty[st], ph ^ 1);
        mbar_expect_tx(&full[st], G30_XBOX);
        const int kx = PR_ISR(i) ? rks * PR_NR + PR_ORD(i) : i0 + PR_ORD(i);
#if PR_F32
        if (PR_ISR(i)) tma2d(xs + st * G30_XBOX, &tmXR, kx * 32, 0, &full[st]); else  // 8 tok x 32 k FP32 = 1 KB = XBOX
#endif
        tma2d(xs + st * G30_XBOX, &tmX, kx * 64, 0, &full[st]);
      }
    }
    return;
  }
  const int wg = warp >> 2, t = tid & 127;
  float d[G30_ND], dr[G30_ND];
#pragma unroll
  for (int j = 0; j < G30_ND; j++) { d[j] = 0.f; dr[j] = 0.f; }
#if G30_ESC
  unsigned cand[2 * G30_ESC], nxt;
#define G30_POP { nxt = cand[0]; _Pragma("unroll") for (int e = 0; e < 2 * G30_ESC - 1; e++) cand[e] = cand[e + 1]; cand[2 * G30_ESC - 1] = ~0u; }
  {
    const int q = t & 3;
#pragma unroll
    for (int rr = 0; rr < 2; rr++)
#pragma unroll
      for (int e = 0; e < G30_ESC; e++) {
        const float2 hv = __ldg(hdr + (size_t)(g * 64 + (t >> 5) * 16 + 8 * rr + ((t & 31) >> 2)) * G30_ESC + e);
        const int col = __float_as_int(hv.x);
        const bool mine = hv.y != 0x1p-15f && ((col >> 1) & 3) == q && ((col >> 6) & 1) == wg;
        cand[rr * G30_ESC + e] = mine ? ((unsigned)col << 18) | ((unsigned)rr << 16) | (__float_as_uint(hv.y) >> 16) : ~0u;
      }
#pragma unroll
    for (int a_ = 0; a_ < 2 * G30_ESC; a_++)
#pragma unroll
      for (int b_ = 2 * G30_ESC - 1; b_ > a_; b_--) {
        const unsigned lo = min(cand[b_ - 1], cand[b_]), hi = max(cand[b_ - 1], cand[b_]);
        cand[b_ - 1] = lo; cand[b_] = hi;
      }
    G30_POP;
    while ((nxt >> 18) < (unsigned)i0 * 64u) G30_POP
  }
#define G30_FIX(a) { const unsigned kend = (unsigned)(ka + 1) * 64u; while ((nxt >> 18) < kend) { esc_patch(a, nxt); G30_POP } }
#else
#define G30_FIX(a)
#endif
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) TLX(ra, 9)
  unsigned A0[16], A1[16];
  int prev = -1;
#define PR_WAIT_ST                                                                                          \
    const int st = ir % G30_NS, ph = (ir / G30_NS) & 1, sw = ir % G30_NW, pw = (ir / G30_NW) & 1;           \
    mbar_wait(&wfull[sw], pw);                                                                              \
    mbar_wait(&full[st], ph);                                                                               \
    const uint8_t* wst = ws + sw * PR_SLOT;
#define PR_MMA(acc, a, pa)                                                                                  \
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                 \
    { const uint32_t b = smem_u32(xs + st * G30_XBOX);                                                      \
    _Pragma("unroll") for (int j = 0; j < 4; j++) wg_mma_rs(acc, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], dsw128(b + j * 32)); } \
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                          \
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");                                          \
    _Pragma("unroll") for (int j = 0; j < 16; j++) asm volatile("" : "+r"(pa[j]));                          \
    if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }   \
    prev = ir;                                                                                              \
    ir += 2;
#define PR_RSTEP(a, pa)                                                                                     \
  {                                                                                                         \
    PR_WAIT_ST                                                                                              \
    _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                         \
      const uint4 v = *(const uint4*)(wst + j * 2048 + t * 16);                                             \
      a[4 * j] = v.x; a[4 * j + 1] = v.y; a[4 * j + 2] = v.z; a[4 * j + 3] = v.w; }                         \
    PR_MMA(dr, a, pa)                                                                                       \
  }
#define G30_DEC(a, j, S0, S1, NW)                            \
    a[4 * j + 0] = decw<0>(S0, NW, 0x4140u);                 \
    a[4 * j + 1] = decw<1>(S0, NW, 0x4342u);                 \
    a[4 * j + 2] = decw<2>(S1, NW, 0x4140u);                 \
    a[4 * j + 3] = decw<3>(S1, NW, 0x4342u);
#define PR_PSTEP(a, pa)                                                                                     \
  {                                                                                                         \
    PR_WAIT_ST                                                                                              \
    const int ka = i0 + PR_ORD(ir);                                                                         \
    const uint4 s01 = *(const uint4*)(wst + t * 16), s23 = *(const uint4*)(wst + 2048 + t * 16);            \
    const uint4 nw = *(const uint4*)(wst + 4096 + t * 16);                                                  \
    G30_DEC(a, 0, s01.x, s01.y, nw.x) G30_DEC(a, 1, s01.z, s01.w, nw.y) G30_DEC(a, 2, s23.x, s23.y, nw.z)   \
    G30_DEC(a, 3, s23.z, s23.w, nw.w) G30_FIX(a)                                                            \
    PR_MMA(d, a, pa)                                                                                        \
  }
  // ---- router stages (PR_NR even: each WG takes PR_NR / 2) ----
  int ir = wg;
#if PR_F32
  // stage = W [64 rows][8 chunks of 4 k] FP32, chunk c of row r at c ^ (r & 7) (host pack) + xr [8 tok][32 k] FP32
  // (TMA SWIZZLE_128B: chunk c of token j at c ^ j); thread: rows rp, rp + 32 x tokens 2 tp, 2 tp + 1; every LDS of
  // a warp = 1 wavefront (W: 8 rows, 8 distinct chunk slots; x: 4 tokens, broadcast); release after a WG barrier
  const int rp = t >> 2, tp = t & 3;
  float f00 = 0.f, f01 = 0.f, f10 = 0.f, f11 = 0.f;
#define PR_FSTEP                                                                                            \
  {                                                                                                         \
    PR_WAIT_ST                                                                                              \
    const uint32_t wf = smem_u32(wst) + rp * 128, xf = smem_u32(xs + st * G30_XBOX) + tp * 256;             \
    _Pragma("unroll") for (int c = 0; c < 8; c++) {                                                         \
      const uint32_t cw = ((c ^ rp) & 7) * 16, cx0 = ((c ^ (2 * tp)) & 7) * 16, cx1 = ((c ^ (2 * tp + 1)) & 7) * 16; \
      const float4 w0 = lds4f(wf + cw), w1 = lds4f(wf + 32 * 128 + cw), x0 = lds4f(xf + cx0), x1 = lds4f(xf + 128 + cx1); \
      f00 = fmaf(w0.x, x0.x, f00); f00 = fmaf(w0.y, x0.y, f00); f00 = fmaf(w0.z, x0.z, f00); f00 = fmaf(w0.w, x0.w, f00); \
      f01 = fmaf(w0.x, x1.x, f01); f01 = fmaf(w0.y, x1.y, f01); f01 = fmaf(w0.z, x1.z, f01); f01 = fmaf(w0.w, x1.w, f01); \
      f10 = fmaf(w1.x, x0.x, f10); f10 = fmaf(w1.y, x0.y, f10); f10 = fmaf(w1.z, x0.z, f10); f10 = fmaf(w1.w, x0.w, f10); \
      f11 = fmaf(w1.x, x1.x, f11); f11 = fmaf(w1.y, x1.y, f11); f11 = fmaf(w1.z, x1.z, f11); f11 = fmaf(w1.w, x1.w, f11); \
    }                                                                                                       \
    asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");                                           \
    if (t == 0) { mbar_arrive(&empty[st]); mbar_arrive(&wempty[sw]); }                                      \
    ir += 2;                                                                                                \
  }
#if PR_IL
#pragma unroll 1
  while (ir < 2 * PR_NR) {
    PR_PSTEP(A0, A1)
    PR_FSTEP
    PR_PSTEP(A1, A0)
    PR_FSTEP
  }
#else
#pragma unroll 1
  while (ir < PR_NR) PR_FSTEP
#endif
  {
    float* rr_ = rred + wg * G30_MTOK * 64;
    rr_[(2 * tp) * 64 + rp] = f00; rr_[(2 * tp + 1) * 64 + rp] = f01;
    rr_[(2 * tp) * 64 + rp + 32] = f10; rr_[(2 * tp + 1) * 64 + rp + 32] = f11;
    asm volatile("bar.arrive 3, 288;" ::: "memory");
  }
#else
#pragma unroll 1
  while (ir < PR_NR) {
    PR_RSTEP(A0, A1)
    PR_RSTEP(A1, A0)
  }
#endif
#if PR_ROUTER && !PR_F32
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));
#pragma unroll
  for (int j = 0; j < G30_ND; j++) asm volatile("" : "+f"(dr[j]));
  // router partial (accumulator layout -> [tok][64 rows]) to rred[wg]; warp 10 sums + stores + counts it
  {
    const int r0 = (warp & 3) * 16 + (lane >> 2), tk0 = 2 * (lane & 3);
    float* rr_ = rred + wg * G30_MTOK * 64;
#pragma unroll
    for (int jn = 0; jn < G30_MTOK / 8; jn++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 2; q++) rr_[(8 * jn + tk0 + q) * 64 + r0 + 8 * h] = dr[4 * jn + 2 * h + q];
    asm volatile("bar.arrive 3, 288;" ::: "memory");
  }
  prev = -1;  // all router stages are complete (wait_group 0): release the last one now
  if (t == 0) {
    const int lr = PR_NR - 2 + wg;
    mbar_arrive(&empty[lr % G30_NS]); mbar_arrive(&wempty[lr % G30_NW]);
  }
#endif
  if (wg == 0) TLM(4)
  // ---- pooler stages ----
#pragma unroll 1
  while (ir < nst) {
    PR_PSTEP(A0, A1)
    if (ir >= nst) break;
    PR_PSTEP(A1, A0)
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 16; j++) asm volatile("" : "+r"(A0[j]), "+r"(A1[j]));
#pragma unroll
  for (int j = 0; j < G30_ND; j++) asm volatile("" : "+f"(d[j]));
  if (prev >= 0 && t == 0) { mbar_arrive(&empty[prev % G30_NS]); mbar_arrive(&wempty[prev % G30_NW]); }
  if (wg == 0) TLM(5)
#if PR_SA
  // rank 1 (both WGs) and rank 0's WG1 push their accumulators into rank 0's rbuf slots; rank 0's WG0 sums
  if (blockIdx.y == 1 || wg == 1) {
    const int slot = blockIdx.y == 1 ? wg : 2;
    uint32_t rb, rm;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rb) : "r"(smem_u32(rbuf + slot * G30_ND * 128)));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rm) : "r"(smem_u32(rbar)));
#pragma unroll
    for (int j = 0; j < G30_ND; j += 4) st_async4(rb + ((j >> 2) * 128 + t) * 16, d[j], d[j + 1], d[j + 2], d[j + 3], rm);
    return;
  }
  asm volatile(
      "{\n .reg .pred p;\n PR32_RW:\n"
      " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
      " @!p bra PR32_RW;\n}\n" :: "r"(smem_u32(rbar)) : "memory");
#pragma unroll
  for (int y = 0; y < 3; y++)
#pragma unroll
    for (int j = 0; j < G30_ND; j += 4) {
      const float4 v = *(const float4*)(rbuf + y * G30_ND * 128 + ((j >> 2) * 128 + t) * 4);
      d[j] += v.x; d[j + 1] += v.y; d[j + 2] += v.z; d[j + 3] += v.w;
    }
#else
  float* red = (float*)xs;
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) {
#pragma unroll
    for (int j = 0; j < G30_ND; j++) red[j * 128 + t] = d[j];
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) return;
#pragma unroll
  for (int j = 0; j < G30_ND; j++) d[j] += red[j * 128 + t];
#if G30_S > 1
  if (blockIdx.y > 0) {
    uint32_t rb, rm;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rb) : "r"(smem_u32(rbuf + (blockIdx.y - 1) * G30_ND * 128)));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rm) : "r"(smem_u32(rbar)));
#pragma unroll
    for (int j = 0; j < G30_ND; j++) asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(rb + (j * 128 + t) * 4), "f"(d[j]) : "memory");
    asm volatile("bar.sync 2, 128;" ::: "memory");
    if (t == 0) asm volatile("fence.acq_rel.cluster;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(rm) : "memory");
    return;
  }
  {
    asm volatile(
        "{\n .reg .pred p;\n PR32_RW:\n"
        " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra PR32_RW;\n}\n" :: "r"(smem_u32(rbar)) : "memory");
#pragma unroll
    for (int y = 0; y < G30_S - 1; y++)
#pragma unroll
      for (int j = 0; j < G30_ND; j++) d[j] += rbuf[(y * G30_ND + j) * 128 + t];
  }
#endif
#endif  // PR_SA
  const int r0 = (warp & 3) * 16 + (lane >> 2);
  const int tok = 2 * (lane & 3);
  __nv_bfloat16* stg = (__nv_bfloat16*)(xs + G30_MTOK * 256);
#pragma unroll
  for (int jn = 0; jn < G30_MTOK / 8; jn++)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int q = 0; q < 2; q++) stg[(8 * jn + tok + q) * 72 + r0 + 8 * h] = __float2bfloat16(d[4 * jn + 2 * h + q] * scale);
  asm volatile("bar.sync 2, 128;" ::: "memory");
#pragma unroll
  for (int q = 0; q < (G30_MTOK + 15) / 16; q++) {
    const int idx = q * 128 + t, tk = idx >> 3, c = idx & 7;
    if (G30_MTOK >= 16 || idx < G30_MTOK * 8) *(uint4*)(out + (size_t)tk * N + g * 64 + c * 8) = *(const uint4*)(stg + tk * 72 + c * 8);
  }
  if (t == 0) TLX(ra, 10)
}

static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
static void mk(CUtensorMap* m, const void* p, uint64_t inner, uint64_t outer, uint32_t b0, uint32_t b1, bool f32 = false) {
  if (!g_enc) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g_enc, 12000, cudaEnableDefault, &q);
  }
  cuuint64_t dims[2] = {inner, outer}, strides[1] = {inner * (f32 ? 4 : 2)};
  cuuint32_t box[2] = {b0, b1}, es[2] = {1, 1};
  CUresult r = g_enc(m, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, es,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "pr32 tensor map encode failed ", (int)r);
}
}  // namespace pr32

// pooled [T][N] BF16 = x [T][K] @ Wp^T (pack30 stages wp, hdr [N][ESC] float2, scale) and the T > 8 router
// (x @ [hi; lo]^T, rw = pack_router stages [8][128][8192]) + routing in one launch. part fp32 [8 RKS T 64] and
// logits fp32 [T][512] scratch; cnt int32 [16] zeros (self-resetting). pdl: launch as a PDL secondary.
void pr32_launch(torch::Tensor x, torch::Tensor xr, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale,
                 torch::Tensor rw, torch::Tensor part, torch::Tensor logits, torch::Tensor cnt, torch::Tensor bias,
                 torch::Tensor ls, torch::Tensor ids, torch::Tensor lids, torch::Tensor wts, double mult, double reps,
                 int64_t local, int64_t pdl, torch::Tensor tl) {
  const int K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.size(0) == G30_MTOK && x.is_contiguous() && K == 8192 && N == 4096 && out.size(0) == G30_MTOK);
  CUtensorMap tx;
  pr32::mk(&tx, x.data_ptr(), K, G30_MTOK, 64, G30_MTOK);
  CUtensorMap txr = tx;
#if PR_F32
  TORCH_CHECK(xr.scalar_type() == torch::kFloat32 && xr.is_contiguous() && xr.size(0) == G30_MTOK && xr.size(1) == K);
  pr32::mk(&txr, xr.data_ptr(), K, G30_MTOK, 32, G30_MTOK, true);
#endif
  static_assert(G30_S == 2, "pr32: pair router reduce needs S = 2");
  const int smem = G30_NS * G30_XBOX + G30_NW * PR_SLOT + 1024 + 3 * G30_MTOK * 64 * 4 + G30_ND * 128 * 4 * PR_RBS + 1024 + 1024;
  // finisher: 64 chunks + 8 x [512] logits below the barriers
  TORCH_CHECK(64 * (G30_MTOK / 4) * PR_RW * 4 <= G30_NS * G30_XBOX + G30_NW * PR_SLOT, "pr32: finisher smem");
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(pr32::pr32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  pr32::RArgs ra{(const uint8_t*)rw.data_ptr(), part.data_ptr<float>(), logits.data_ptr<float>(), cnt.data_ptr<int>(),
                 bias.data_ptr<float>(), ls.data_ptr<int>(), ids.data_ptr<int>(), lids.data_ptr<int>(),
                 wts.data_ptr<float>(), (float)mult, (float)reps, (int)local, (unsigned long long*)tl.data_ptr()};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64 + 2, G30_S);
  cfg.blockDim = dim3(352);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = 1; at[1].val.clusterDim.y = G30_S; at[1].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = G30_S > 1 ? 2 : 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, pr32::pr32_kernel, tx, txr, (const uint8_t*)wp.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
                                 (const float2*)hdr.data_ptr(), N, K / 64, (float)scale, ra) == cudaSuccess, "pr32 launch failed");
}
