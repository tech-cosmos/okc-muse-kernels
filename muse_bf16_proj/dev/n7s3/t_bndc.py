# correctness: bndc vs layer _boundary (T=256 and 128, BGAIN off/on): s_out and h
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/r-proj/muse_decode_layer"))
import submission as L, bndc as B
torch.manual_seed(1)
N = 8192
CS = int(os.environ.get('CS', '8'))
for T in (256, 128):
    for bg in (False, True):
        a = (torch.randn(T, N, device="cuda") * 3).bfloat16(); s = torch.randn(T, N, device="cuda") * 5
        al, be, g1, g2 = [torch.rand(N, device="cuda") + 0.5 for _ in range(4)]
        so0, so1 = torch.empty_like(s), torch.empty_like(s); h0, h1 = torch.empty_like(a), torch.empty_like(a)
        L._boundary[(T,)](a, s, al, be, g2, g1, so0, h0, h0, NB=N, N=N, BGAIN=bg, H32=False, EPS_POST=L.POST_NORM_EPS,
                          EPS=L.RMS_EPS, PDL=False, RB=False, num_warps=16, SPL=1)
        (B.lib().bndp(a, s, al, be, g2, g1, so1, h1, L.POST_NORM_EPS, L.RMS_EPS, int(os.environ['RPC']), bg) if 'RPC' in os.environ else B.lib().bndc(a, s, al, be, g2, g1, so1, h1, L.POST_NORM_EPS, L.RMS_EPS, CS, bg))
        torch.cuda.synchronize()
        ds = ((so0 - so1).abs().amax(1) / so0.pow(2).mean(1).sqrt()).max().item()
        dh = (h0.float() != h1.float()).float().mean().item()
        dhm = ((h0.float() - h1.float()).abs() / h0.float().abs().clamp_min(1e-3)).max().item()
        print(f"T={T} bg={bg}: s_out max rel {ds:.2e}, h differ {dh*100:.3f}% (max rel {dhm:.2e})")
