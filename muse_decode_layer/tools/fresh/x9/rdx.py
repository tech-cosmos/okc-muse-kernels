import sys
OLD = """#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bs, o);
            if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; bs = os; }
        }
"""
NEW = """        {  // fresh x9: warp argmax by redux.sync (max key, then lowest index among the ties), not 15 dependent shuffles
            unsigned u = __float_as_uint(bk + 0.f);  // -0 -> +0: float == ties stay ties
            u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // order-preserving float -> uint
            const unsigned m = __reduce_max_sync(0xffffffffu, u);
            bi = (int)__reduce_min_sync(0xffffffffu, u == m ? (unsigned)bi : 0xffffffffu);
            bs = __shfl_sync(0xffffffffu, bs, bi & 31);
        }
"""
p, q = sys.argv[1], sys.argv[2]
s = open(p).read()
n = s.count(OLD)
print("sites", n)
i = s.find("const float lg = ")  # pr1's routing only: the first site after pr1's logits load
j = s.find(OLD, i)
assert j > 0
s = s[:j] + NEW + s[j + len(OLD):]
open(q, "w").write(s)
