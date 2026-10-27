#define PR_RST 8192                 // router stage: 64 rows x 64 k raw BF16 in A-fragment order [4 j][128 thr][16 B]
#define PR_SLOT (PR_RST > G30_WST ? PR_RST : G30_WST)
#define PR_RKS (8 * G30_S)          // router K slices per 64-row group (8 groups x RKS = 64 S CTAs)
#ifndef PR_ROUTE
#define PR_ROUTE 1
#endif
#ifndef PR_TICKET
#define PR_TICKET 1
#endif
#ifndef PR_ROUTER
#define PR_ROUTER 1
#endif
#define PR_NR (PR_ROUTER ? 128 / PR_RKS : 0)  // router stages per CTA
constexpr int E = 256, TOPK = 8;

struct RArgs {
  const uint8_t* rw; float* part; float* logits; int* cnt; const float* bias; const int* ls;
  int* ids; int* lids; float* wts; float mult, reps; int local; unsigned long long* tl;
};
__device__ __forceinline__ unsigned long long gtime() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#ifdef PR_TL
#define TLX(a_, k) a_.tl[(blockIdx.x * G30_S + blockIdx.y) * 8 + (k)] = gtime();
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
    const float l = lg[e] + lg[E + e];  // hi + lo
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

// finishers (clusters x = 0, 1: 4 CTAs on the 4 SMs the 128 streaming CTAs leave free): f = 2 x + y routes tokens
// 8 f .. 8 f + 7. Waits for the 64 pair partials, pulls its 8-token chunks of all of them (64 x 2 KB) in one bulk shot,
// sums them in pair-slice order (deterministic), routes one token per warp.
__device__ __forceinline__ void finisher(const RArgs& a, uint8_t* sm, uint64_t* fb, float* bs_, float* lgb) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, f = 2 * blockIdx.x + blockIdx.y;
  constexpr int TPF = G30_MTOK / 4, CH = TPF * 64 * 4, NP = 8;  // tokens per finisher, chunk bytes, pair slices per rg
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
      bulk1d(sm + i * CH, (const uint8_t*)a.part + ((size_t)i * G30_MTOK + TPF * f) * 64 * 4, CH, fb);
  }
  __syncthreads();
  mbar_wait(fb, 0);
  if (tid == 0) TLX(a, 2)
  if (warp < TPF) {
    const float* c = (const float*)sm;
    float* lg = lgb + warp * 2 * E;  // this warp's token: [512] logits
#pragma unroll
    for (int i = 0; i < 2 * E / 32; i++) {
      const int row = lane + 32 * i, rg = row >> 6, r = row & 63;
      float s = 0.f;
#pragma unroll
      for (int p = 0; p < NP; p++) s += c[((rg * NP + p) * TPF + warp) * 64 + r];
      lg[row] = s;
    }
    __syncwarp();
    route_token(a, lg, bs_, TPF * f + warp, lane, ls);
#pragma unroll
    for (int i = 0; i < 2 * E / 32; i++) __stcg(a.logits + (TPF * f + warp) * 2 * E + lane + 32 * i, lg[lane + 32 * i]);
    if (tid == 0) TLX(a, 7)
  }
}

// grid (64, S) in (1, S, 1) clusters: CTA (g, y) = pooler rows 64 g.., K part y (S = 2: halves, DSMEM sum into rank 0)
// FIRST does router item r = S g + y: router rows (hi 0..255 | lo 256..511) 64 (r / RKS).., K slice r % RKS (NR stages).
__global__ void __launch_bounds__(352, 1)
pr32_kernel(const __grid_constant__ CUtensorMap tmX, const uint8_t* __restrict__ wp, __nv_bfloat16* __restrict__ out,
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
  if (tid == 0) {
    for (int i = 0; i < G30_NS; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 1); }
    for (int i = 0; i < G30_NW; i++) { mbar_init(&wfull[i], 1); mbar_init(&wempty[i], 1); }
    mbar_init(rbar, G30_S > 1 ? G30_S - 1 : 1);
    mbar_init(pbar, 1);
    mbar_init(fbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#if G30_S > 1
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#else
  __syncthreads();
#endif
  if (g < 0) { finisher(ra, sm, fbar, rbuf + G30_ND * 128 * (G30_S - 1), rred); return; }
  if (warp == 10) {  // router partial: (WG0 + WG1) [+ rank 1's via DSMEM] -> part[rg][pair slice], then count it
    asm volatile("bar.sync 3, 288;" ::: "memory");
    if (blockIdx.y == 1) {
      uint32_t rb, rm;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rb) : "r"(smem_u32(rpair)));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rm) : "r"(smem_u32(pbar)));
      for (int idx = lane * 4; idx < G30_MTOK * 64; idx += 128) {
        const float4 u = *(const float4*)(rred + idx), v = *(const float4*)(rred + G30_MTOK * 64 + idx);
        asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(rb + idx * 4), "f"(u.x + v.x), "f"(u.y + v.y),
                     "f"(u.z + v.z), "f"(u.w + v.w) : "memory");
      }
      __syncwarp();
      if (lane == 0) asm volatile("fence.acq_rel.cluster;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(rm) : "memory");
      return;
    }
    asm volatile(
        "{\n .reg .pred p;\n PR32_PW:\n"
        " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra PR32_PW;\n}\n" :: "r"(smem_u32(pbar)) : "memory");
    float* pp = ra.part + (size_t)(rg * 8 + (g & 7)) * G30_MTOK * 64;
    if (lane == 0) TLX(ra, 0)
    for (int idx = lane * 4; idx < G30_MTOK * 64; idx += 128) {
      const float4 u = *(const float4*)(rred + idx), v = *(const float4*)(rred + G30_MTOK * 64 + idx);
      const float4 w = *(const float4*)(rpair + idx);
      __stcg((float4*)(pp + idx), make_float4(u.x + v.x + w.x, u.y + v.y + w.y, u.z + v.z + w.z, u.w + v.w + w.w));
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(ra.cnt) : "memory");
    if (lane == 0) TLX(ra, 1)
    return;
  }
  if (warp == 8) {  // W: router stages then pooler stages (weights only: before griddepcontrol.wait)
    if (lane == 0) {
      const uint8_t* rsrc = ra.rw + ((size_t)rg * 128 + (size_t)rks * PR_NR) * PR_RST;
      const uint8_t* src = wp + ((size_t)g * KS + i0) * G30_WST;
      for (int i = 0; i < nst; i++) {
        const int st = i % G30_NW, ph = (i / G30_NW) & 1;
        mbar_wait(&wempty[st], ph ^ 1);
        if (i < PR_NR) {
          mbar_expect_tx(&wfull[st], PR_RST);
          bulk1d(ws + st * PR_SLOT, rsrc + (size_t)i * PR_RST, PR_RST, &wfull[st]);
        } else {
          mbar_expect_tx(&wfull[st], G30_WST);
          bulk1d(ws + st * PR_SLOT, src + (size_t)(i - PR_NR) * G30_WST, G30_WST, &wfull[st]);
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
        const int kx = i < PR_NR ? rks * PR_NR + i : i0 + i - PR_NR;
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
    const int ka = i0 + ir - PR_NR;                                                                         \
    const uint4 s01 = *(const uint4*)(wst + t * 16), s23 = *(const uint4*)(wst + 2048 + t * 16);            \
    const uint4 nw = *(const uint4*)(wst + 4096 + t * 16);                                                  \
    G30_DEC(a, 0, s01.x, s01.y, nw.x) G30_DEC(a, 1, s01.z, s01.w, nw.y) G30_DEC(a, 2, s23.x, s23.y, nw.z)   \
    G30_DEC(a, 3, s23.z, s23.w, nw.w) G30_FIX(a)                                                            \
    PR_MMA(d, a, pa)                                                                                        \
  }
  // ---- router stages (PR_NR even: each WG takes PR_NR / 2) ----
  int ir = wg;
#pragma unroll 1
  while (ir < PR_NR) {
    PR_RSTEP(A0, A1)
    PR_RSTEP(A1, A0)
  }
#if PR_ROUTER
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
  for (int q = 0; q < G30_MTOK / 16; q++) {
    const int idx = q * 128 + t, tk = idx >> 3, c = idx & 7;
    *(uint4*)(out + (size_t)tk * N + g * 64 + c * 8) = *(const uint4*)(stg + tk * 72 + c * 8);
  }
}

static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
static void mk(CUtensorMap* m, const void* p, uint64_t inner, uint64_t outer, uint32_t b0, uint32_t b1) {
  if (!g_enc) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g_enc, 12000, cudaEnableDefault, &q);
  }
  cuuint64_t dims[2] = {inner, outer}, strides[1] = {inner * 2};
  cuuint32_t box[2] = {b0, b1}, es[2] = {1, 1};
  CUresult r = g_enc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, es,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "pr32 tensor map encode failed ", (int)r);
}
}  // namespace pr32

// pooled [T][N] BF16 = x [T][K] @ Wp^T (pack30 stages wp, hdr [N][ESC] float2, scale) and the T > 8 router
// (x @ [hi; lo]^T, rw = pack_router stages [8][128][8192]) + routing in one launch. part fp32 [8 RKS T 64] and
// logits fp32 [T][512] scratch; cnt int32 [16] zeros (self-resetting). pdl: launch as a PDL secondary.
void pr32_launch(torch::Tensor x, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale,
                 torch::Tensor rw, torch::Tensor part, torch::Tensor logits, torch::Tensor cnt, torch::Tensor bias,
                 torch::Tensor ls, torch::Tensor ids, torch::Tensor lids, torch::Tensor wts, double mult, double reps,
                 int64_t local, int64_t pdl, torch::Tensor tl) {
  const int K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.size(0) == G30_MTOK && x.is_contiguous() && K == 8192 && N == 4096 && out.size(0) == G30_MTOK);
  CUtensorMap tx;
  pr32::mk(&tx, x.data_ptr(), K, G30_MTOK, 64, G30_MTOK);
  static_assert(G30_S == 2, "pr32: pair router reduce needs S = 2");
  const int smem = G30_NS * G30_XBOX + G30_NW * PR_SLOT + 1024 + 3 * G30_MTOK * 64 * 4 + G30_ND * 128 * 4 * (G30_S - 1) + 1024 + 1024;
  // finisher: 64 chunks + 8 x [512] logits below the barriers
  TORCH_CHECK(64 * (G30_MTOK / 4) * 256 <= G30_NS * G30_XBOX + G30_NW * PR_SLOT, "pr32: finisher smem");
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
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, pr32::pr32_kernel, tx, (const uint8_t*)wp.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
                                 (const float2*)hdr.data_ptr(), N, K / 64, (float)scale, ra) == cudaSuccess, "pr32 launch failed");
}
