import sys
p = sys.argv[1]; s = open(p).read()
i0 = s.index("__global__ void __launch_bounds__(384, 1) moe_twm(")
i1 = s.index("// ---- s29 moe_hot", i0)
t = s[i0:i1]
def rep(a, b):
    global t
    assert t.count(a) == 1, a[:90]
    t = t.replace(a, b)
rep("__nv_bfloat16* __restrict__ Y, int* __restrict__ tick, float* __restrict__ part, int nsplit, int hx = 0) {",
    """__nv_bfloat16* __restrict__ Y, int* __restrict__ tick, float* __restrict__ part, int nsplit, int hx = 0,
    unsigned long long* __restrict__ tws = nullptr, int stw = 0) {
  // n5s17 TWS (stw > 0): work stealing in the gemm2 tail. The last NV = stw & 1023 items, if light gemm2 (RT 4), are
  // 'stealable': the owner claims its K chunks in batches of B ((stw >> 16) & 63, default 8) from a per-item state word
  // (lo | hi << 8 | pieces << 16 | done << 24); a CTA that finds no item left steals the back half of the stealable
  // item with the most unclaimed chunks (>= MS = (stw >> 10) & 63, default 12) and runs it as a piece. Pieces store
  // FP32 partials in slots (owner of item v: slot v; thieves: NV + blockIdx * TSL + k); the last finisher sums them in
  // piece order and runs the epilogue. Words are zero between calls (finisher / unstolen owner reset them).
  // tws = [NV] state words, then int smap[NV][8]; partial slots at part + 67 * 32768.""")
rep("  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;",
    """  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;
  constexpr int TSL = 4;
  const int SNV = stw & 1023, SMS = ((stw >> 10) & 63) ? (stw >> 10) & 63 : 12, SB_ = ((stw >> 16) & 63) ? (stw >> 16) & 63 : 8;
  const int sbase = nitems - SNV;  // stealable items: [sbase, nitems) (needs ns == 0: host passes nsplit 0 with stw)
  int* smap = reinterpret_cast<int*>(tws + SNV);
  float* spart = part + (size_t)67 * 32768;
  __shared__ int s_last[NSTG], s_pc[NSTG];""")
# producer: thief loop at the end + owner batch claims
rep("""      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }""", """      if (item >= nitems) {
        if (SNV) {  // n5s17 TWS thief loop (the slot st is free: waited above)
          int nst = 0;
          bool ready = true;
          while (nst < TSL) {
            int best = -1, brem = SMS - 1;
            unsigned long long bw = 0;
#pragma unroll 8
            for (int v = 0; v < SNV; ++v) {
              unsigned long long w;
              asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(w) : "l"(tws + v) : "memory");
              const int rem = (int)((w >> 8) & 255) - (int)(w & 255);
              if (rem > brem && ((w >> 16) & 255) < 8) { brem = rem; best = v; bw = w; }
            }
            if (best < 0) break;
            const int lo = bw & 255, hi = (bw >> 8) & 255, np = (bw >> 16) & 255;
            const int mid = lo + (hi - lo + 1) / 2;
            const unsigned long long nw = (bw & ~(255ull << 8) & ~(255ull << 16)) | ((unsigned long long)mid << 8) |
                                          ((unsigned long long)(np + 1) << 16);
            if (atomicCAS(tws + best, bw, nw) != bw) continue;
            const int sit = sbase + best, slot = SNV + blockIdx.x * TSL + nst++;
            int g2, a, p, rb, h, j;
            decode(sit, g2, a, p, rb, h, j);
            const int e = s_e[a];
            const int off = s_off[e] + p * 32, nv = min(32, s_off[e + 1] - off);
            const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : 2;
            const char* wsrc = W2 + ((size_t)e * CN::NRB + rb) * 64 * CG::WB;
            const char* ssrc = S2 + ((size_t)e * CN::NRB + rb) * 64 * CG::SB;
            const int bytes = CG::WB + CG::SB + (1024 << mi);
            asm volatile("fence.proxy.async.global;" ::: "memory");  // done[e] was acquired by the owner already
            for (int kc = mid; kc < hi; ++kc, ++i) {
              if (!ready && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
              ready = false;
              char* sb = smem + st * STRIDE;
              if (kc == mid) s_item[st] = sit | (1 << 30);
              s_last[st] = kc == hi - 1 ? 3 : 0;
              s_pc[st] = np | (slot << 4);
              mbar_expect_tx(&full[st], bytes);
              tma2d(sb, &maps2.m[mi], kc * 64, off, &full[st]);
              bulk_g2s(sb + CG::XB, wsrc + (size_t)kc * CG::WB, CG::WB, &full[st]);
              bulk_g2s(sb + CG::XB + CG::WB, ssrc + (size_t)kc * CG::SB, CG::SB, &full[st]);
              if (++st == NSTG) { st = 0; ph ^= 1; }
            }
          }
          if (!ready && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        }
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }""")
rep("""      for (int kc = k0; kc < k1; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (kc == k0) s_item[st] = item;""", """      // n5s17 TWS owner: claim K chunks in batches; the last stage tells the consumers whether the item was stolen from
      const bool stl = SNV && item >= sbase && g2 && cl == 0 && h < 0;
      unsigned long long* sw = stl ? tws + (item - sbase) : nullptr;
      int kend = k1;
      if (stl) {
        kend = SB_;
        atomicExch(sw, (unsigned long long)SB_ | (64ull << 8) | (1ull << 16));
      }
      for (int kc = k0; kc < kend; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (kc == k0) s_item[st] = stl ? item | (1 << 30) : item;
        if (stl) {
          int lst = 0;
          if (kc == kend - 1) {
            const unsigned long long o = atomicAdd(sw, (unsigned long long)SB_);
            const int lo = o & 255, hi = (o >> 8) & 255, np = (o >> 16) & 255;
            if (lo < hi) kend = min(lo + SB_, hi);  // next batch [lo, kend) follows contiguously? no: lo == kend here
            else {
              lst = np > 1 ? 2 : 1;
              if (np == 1) atomicExch(sw, 0ull);  // unstolen: nobody else will touch the word
            }
          }
          s_last[st] = lst;
          s_pc[st] = (item - sbase) << 4;
        }""")
# consumer side
rep("""    if (item < 0) break;
    int g2, a, p, rb, h, j;
    decode(item, g2, a, p, rb, h, j);""", """    if (item < 0) break;
    const int dyn = (item >> 30) & 1;
    int g2, a, p, rb, h, j;
    decode(item & ~(1 << 30), g2, a, p, rb, h, j);
    const int sv = (item & ~(1 << 30)) - sbase;""")
rep("""                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384)""",
    """                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384, \\
                                                        dyn ? s_last : nullptr, s_pc, tws + sv, smap + sv * 8, spart)""")
open("/tmp/pj17/twm_new.cu", "w").write(t)
s = s[:i0] + t + s[i1:]
open(p, "w").write(s)
print("ok")
