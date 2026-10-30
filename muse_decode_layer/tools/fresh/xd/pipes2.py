# fresh x1 (lead 07:20Z expert-kernel lane): sm_90 issue/pipe rates of the dq instruction mix (H100 = H200 per SM clock).
# Each test: 1 CTA, W warps (W/4 per SMSP), 16 independent chains per thread, NIT iterations; prints cycles per warp-instr per SMSP.
import torch, os, fcntl
from torch.utils.cpp_extension import load_inline
OPS = {
 'lop3': 'lop3.b32 %0, %0, %1, 0x81c081c0, 0xc0;',
 'shf':  'shf.l.wrap.b32 %0, %0, %0, 10;',
 'imad': 'mad.lo.u32 %0, %0, 16648, %1;',
 'prmt': 'prmt.b32 %0, %0, %1, 0x5140;',
 'hmul2': 'mul.rn.bf16x2 %0, %0, %1;',
 'hfma2': 'fma.rn.bf16x2 %0, %0, %1, %1;',
 'iadd3': 'add.u32 %0, %0, %1;',
 'fmul': 'mul.rn.f32 %0, %0, %1;',
 'imadw': '{.reg .b64 t; mul.wide.u32 t, %0, %1; .reg .b32 l, h; mov.b64 {l, h}, t; lop3.b32 %0, l, h, %1, 0xe8;}',
 'imadw_only': '{.reg .b64 t; mul.wide.u32 t, %0, %1; .reg .b32 l, h; mov.b64 {l, h}, t; xor.b32 %0, l, h;}',
}
def kern(name, seq):
    body = "\n".join(f'asm volatile("{OPS[o]}" : "+r"(v[{i % 16}]) : "r"(c));' for i, o in enumerate(seq * (16 // len(seq) if len(seq) < 16 else 1)))
    return f"""
__global__ void k_{name}(unsigned long long* out, unsigned* sink, int nit, unsigned c) {{
  unsigned v[16];
  for (int i = 0; i < 16; ++i) v[i] = threadIdx.x * 7 + i;
  __syncthreads();
  unsigned long long t0 = clock64();
  for (int it = 0; it < nit; ++it) {{
    #pragma unroll
    for (int u = 0; u < 4; ++u) {{ {body} }}
  }}
  unsigned long long t1 = clock64();
  unsigned s = 0; for (int i = 0; i < 16; ++i) s ^= v[i];
  sink[threadIdx.x] = s;
  if (threadIdx.x == 0) out[0] = t1 - t0;
}}"""
TESTS = {
 'lop3': ['lop3'], 'shf': ['shf'], 'imad': ['imad'], 'prmt': ['prmt'], 'hmul2': ['hmul2'], 'hfma2': ['hfma2'],
 'iadd3': ['iadd3'], 'fmul': ['fmul'],
 'lop3+hmul2': ['lop3', 'hmul2'], 'lop3+imad': ['lop3', 'imad'], 'shf+hmul2': ['shf', 'hmul2'],
 'lop3+prmt': ['lop3', 'prmt'], 'prmt+hmul2': ['prmt', 'hmul2'], 'lop3+fmul': ['lop3', 'fmul'], 'lop3+iadd3': ['lop3', 'iadd3'],
 'imad+hmul2': ['imad', 'hmul2'], 'imadw': ['imadw'], 'imadw_only': ['imadw_only'],
 'dqmix': ['lop3', 'shf', 'lop3', 'shf', 'lop3', 'lop3', 'imad', 'lop3', 'hmul2', 'hmul2', 'hmul2', 'hmul2', 'lop3', 'shf', 'lop3', 'hmul2'],
}
src = "\n".join(kern(n.replace('+', '_'), s) for n, s in TESTS.items())
decl = "\n".join(f"void r_{n.replace('+','_')}(torch::Tensor o, torch::Tensor s, int64_t nit, int64_t w);" for n in TESTS)
wr = "\n".join(f"void r_{n.replace('+','_')}(torch::Tensor o, torch::Tensor s, int64_t nit, int64_t w) {{ k_{n.replace('+','_')}<<<1, w * 32>>>((unsigned long long*)o.data_ptr(), (unsigned*)s.data_ptr(), (int)nit, 0x3f803f80u); }}" for n in TESTS)
lib = load_inline("fresh_pipes2", cpp_sources=decl, cuda_sources="#include <torch/extension.h>\n" + src + wr,
                  functions=[f"r_{n.replace('+','_')}" for n in TESTS], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
o = torch.zeros(1, dtype=torch.int64, device='cuda'); s = torch.zeros(1024, dtype=torch.int32, device='cuda')
NIT = 2000
for n in [t for t in TESTS if t.startswith('imadw') or t in ('lop3','dqmix')]:
    f = getattr(lib, f"r_{n.replace('+','_')}")
    row = []
    for w in (4, 8, 16, 32):
        f(o, s, 10, w); torch.cuda.synchronize()
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX); f(o, s, NIT, w); torch.cuda.synchronize()
        ninst = NIT * 4 * 16 * (w // 4)  # warp-instrs per SMSP
        row.append(o.item() / ninst)
    print(f"{n:12s} cyc per warp-instr per SMSP @ 1/2/4/8 warps per SMSP: " + " ".join(f"{x:5.2f}" for x in row), flush=True)
