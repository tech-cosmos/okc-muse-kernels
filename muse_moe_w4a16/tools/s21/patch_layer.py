# apply the s21 TK2F_L2H change to a layer moe_src.py (layer's tk2f_produce text, no FP xrdy):
#   python patch_layer.py <moe_src.py> <tk2f mode> [<twm mode 0|1>]
import sys
p, mode = sys.argv[1], int(sys.argv[2]); twm = int(sys.argv[3]) if len(sys.argv) > 3 else 0; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]; s = s.replace(old, new)
anchor = '''__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
'''
rep(anchor, anchor + '''__device__ __forceinline__ void bulk_g2s_h(void* dst, const void* src, int bytes, uint64_t* bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol) : "memory");
}
''')
rep('''#define TK2_EXP 0 ''', f'''#define TK2F_L2H {mode}       // s21: tk2f weight-copy L2 hints: 1 = evict_last for non-final 8-token groups of an expert, evict_first else; 2 = evict_last non-final only; 3 = evict_first final only; 4 = evict_last every group of a multi-group expert, evict_first else
#define TK2_EXP 0 ''')
rep('''  constexpr int RPE = NRT / 2, NQ = 64 / (NC * KPS), SB = NC * KPS * 1152, KB = NC * KPS;
  int a = 0;
  for (int R = r0; R < r1; ++R) {
    while (s_rs[a + 1] <= R) ++a;
    const int e = s_e[a];
    const int lr = (R - s_rs[a]) & (RPE - 1);
    const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
    const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
    for (int q = 0; q < NQ; ++q, ++i) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      char* sbase = reinterpret_cast<char*>(smem) + st * SB;
      mbar_expect_tx(&full[st], SB);
      bulk_g2s(sbase, src + q * KB * 32, KB * 512, &full[st]);
      bulk_g2s(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st]);
      bulk_g2s(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
    }
  }''', '''  constexpr int RPE = NRT / 2, NQ = 64 / (NC * KPS), SB = NC * KPS * 1152, KB = NC * KPS;
  int a = 0;
#if TK2F_L2H
  // s21: an expert with > 8 tokens is streamed once per 8-token group; the groups' rows sit 256 rows apart (other
  // CTAs, other times), so without hints the re-read often comes from HBM again (H100 ncu d16 s7: +12% DRAM bytes).
  uint64_t pol_last, pol_first;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol_last));
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol_first));
#endif
  for (int R = r0; R < r1; ++R) {
    while (s_rs[a + 1] <= R) ++a;
    const int e = s_e[a];
    const int lr = (R - s_rs[a]) & (RPE - 1);
    const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
    const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
#if TK2F_L2H
    // mode 1-3: nfin = a later group of this expert re-reads this row; mode 4: any group of a multi-group expert
    // (the other group's CTA may read the same row first)
    const bool nfin = TK2F_L2H == 4 ? s_rs[a + 1] - s_rs[a] > RPE : R + RPE < s_rs[a + 1];
    const int hk = nfin ? (TK2F_L2H == 3 ? 0 : 1) : (TK2F_L2H == 2 ? 0 : 2);  // 0 none, 1 evict_last, 2 evict_first
    const uint64_t pol = hk == 1 ? pol_last : pol_first;
#endif
    for (int q = 0; q < NQ; ++q, ++i) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      char* sbase = reinterpret_cast<char*>(smem) + st * SB;
      mbar_expect_tx(&full[st], SB);
#if TK2F_L2H
      if (hk) {
        bulk_g2s_h(sbase, src + q * KB * 32, KB * 512, &full[st], pol);
        bulk_g2s_h(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st], pol);
        bulk_g2s_h(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st], pol);
      } else
#endif
      {
        bulk_g2s(sbase, src + q * KB * 32, KB * 512, &full[st]);
        bulk_g2s(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st]);
        bulk_g2s(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st]);
      }
      if (++st == NSTG) { st = 0; ph ^= 1; }
    }
  }''')
if twm:
    rep('''        tma2d(sb, map, kc * 64, off, &full[st]);
        bulk_g2s(sb + xo, wsrc + (size_t)kc * WB, wbytes, &full[st]);
        bulk_g2s(sb + xo + wbytes, ssrc + (size_t)kc * SB, sbytes, &full[st]);
''', '''        tma2d(sb, map, kc * 64, off, &full[st]);
        {  // s21: evict_last if a later pass of this expert re-reads these weights, else evict_first
          uint64_t pol;
          if (off + PS < s_off[e + 1]) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
          else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
          bulk_g2s_h(sb + xo, wsrc + (size_t)kc * WB, wbytes, &full[st], pol);
          bulk_g2s_h(sb + xo + wbytes, ssrc + (size_t)kc * SB, sbytes, &full[st], pol);
        }
''')
open(p, 'w').write(s)
