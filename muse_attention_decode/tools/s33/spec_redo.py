"""s33: verify 1 full with spec: the redo path (page table wider than seq + L) and the exact width, vs reference.py."""
import sys, torch
sys.path.insert(0, ".")
import reference, submission
for pad, fill in ((0, None), (5, -1), (300, 0), (64, -1)):
    d = reference.generate_input(seed=7 + pad, batch=1, ctx=80000, heads="tp8", layer="full", q_len=8, page=1,
                                 dist="coherent")
    if pad:
        pt = d["page_table"]
        ext = torch.full((pt.shape[0], pad), fill, dtype=pt.dtype, device=pt.device)
        d = {**d, "page_table": torch.cat([pt, ext], 1).contiguous()}
    out = submission.kernel_fn(d)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(d, reference.collect(d, out))
    print(f"pad {pad:4d} fill {fill}: {'PASS' if ok else 'FAIL'} {msg[:110]}", flush=True)
