"""s99 (lead 13:30Z item 2): T=128/256 GEMM over 12-bit packed BF16 weights, Triton prototype, H100 time vs cuBLAS.
Layout (prototype, not proj's DEC3): sm plane [N, K] uint8 (sign | mantissa), code plane [N, K/2] uint8 where, per
BK-tile, byte j holds code k0+j (low nibble) and k0+BK/2+j (high nibble). Decoded BF16 = sign | (112 | c) << 7 | mant
(= w / scale); out = scale * acc (+ escape fixups, not in this bound: weights here have the escapes' decoded value).
python dev/s99_g12t.py [T ...]"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton, triton.language as tl
import bench


@triton.jit
def _dec(s, c):
    b = ((s.to(tl.int32) & 0x80) << 8) | ((c.to(tl.int32) | 112) << 7) | (s.to(tl.int32) & 0x7F)
    return b.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def g12t(x, sm, cd, out, T, N, K, scale, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr,
         MODE: tl.constexpr = 0):
    # MODE 0 = full; 1 = plain BF16 W (sm points at the decoded BF16 W); 2 = x read once (no x re-read, diag)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    H: tl.constexpr = BK // 2
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rh = tl.arange(0, H)
    acc = tl.zeros((BM, BN), tl.float32)
    if MODE == 2:
        x0 = tl.load(x + rm[:, None] * K + rh[None, :])
        x1 = tl.load(x + rm[:, None] * K + H + rh[None, :])
    for k in tl.range(0, K, BK, num_stages=NS):
        if MODE != 2:
            x0 = tl.load(x + rm[:, None] * K + k + rh[None, :])
            x1 = tl.load(x + rm[:, None] * K + k + H + rh[None, :])
        s0 = tl.load(sm + rn[:, None] * K + k + rh[None, :])
        s1 = tl.load(sm + rn[:, None] * K + k + H + rh[None, :])
        if MODE == 1:
            acc = tl.dot(x0, tl.trans(s0), acc)
            acc = tl.dot(x1, tl.trans(s1), acc)
        else:
            c = tl.load(cd + rn[:, None] * (K // 2) + k // 2 + rh[None, :])
            acc = tl.dot(x0, tl.trans(_dec(s0, c & 15)), acc)
            acc = tl.dot(x1, tl.trans(_dec(s1, c >> 4)), acc)
    tl.store(out + rm[:, None] * N + rn[None, :], (acc * scale).to(tl.bfloat16))


def pack(w, BK):
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    base = int(e.max()) - 15
    code = (e - base).clamp(min=0)
    sm = (((b >> 8) & 0x80) | (b & 0x7F)).to(torch.uint8)
    cv = code.view(n, k // BK, 2, BK // 2)
    cd = (cv[:, :, 0] | (cv[:, :, 1] << 4)).to(torch.uint8).reshape(n, k // 2).contiguous()
    dec = (((sm.int() & 0x80) << 8) | ((code + base) << 7) | (sm.int() & 0x7F))
    dec = torch.where(dec >= 32768, dec - 65536, dec).to(torch.int16).view(torch.bfloat16)
    return sm.contiguous(), cd, 2.0 ** (base - 112), dec


def run(T, name, N, K, cfgs):
    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    x = torch.randn(T, K, device="cuda").bfloat16()
    o = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)
    e0 = bench.empty_cost()
    tc = bench.time_graph(lambda: torch.mm(x, w.t(), out=o))
    print(f"{name} T={T} N={N} K={K}: cuBLAS {tc:.1f} us ({N*K*2/tc/1e6:.2f} TB/s bf16)", flush=True)
    packs = {}
    res = []
    for BM, BN, BK, NS, nw in cfgs:
        if BK not in packs:
            packs[BK] = pack(w, BK)
        sm, cd, scale, dec = packs[BK]
        ref = (x.float() @ dec.float().t()).bfloat16()
        grid = (N // BN, T // BM)
        MODE = int(os.environ.get("MODE", "0"))
        if MODE == 1:
            sm, scale, ref = dec, 1.0, (x.float() @ dec.float().t()).bfloat16()
        f = lambda: g12t[grid](x, sm, cd, o, T, N, K, scale, BM=BM, BN=BN, BK=BK, NS=NS, num_warps=nw, MODE=MODE)  # noqa: E731
        try:
            f()
            torch.cuda.synchronize()
            err = ((o.float() - ref.float()).norm() / ref.float().norm()).item()
            t = bench.time_graph(f) - e0
        except Exception as ex:  # noqa: BLE001
            print(f"  {BM},{BN},{BK},{NS},{nw}: fail {str(ex)[:100]}")
            continue
        res.append((t, (BM, BN, BK, NS, nw)))
        print(f"  BM{BM} BN{BN} BK{BK} NS{NS} W{nw} grid{grid}: {t:.1f} us ({N*K*1.5/t/1e6:.2f} TB/s 12-bit) err {err:.1e}", flush=True)
    del w, x, o, packs
    torch.cuda.empty_cache()
    return sorted(res)[:3]


if __name__ == "__main__":
    Ts = [int(a) for a in sys.argv[1:]] or [128]
    for T in Ts:
        BMs = [128] if T == 128 else [128, 256]
        cfgs = [c for c in itertools.product(BMs, (32, 64), (128,), (3, 4), (4, 8))]
        for name, N, K in (("pooler", 4096, 8192), ("unpool", 8192, 4096)):
            print(run(T, name, N, K, cfgs))
