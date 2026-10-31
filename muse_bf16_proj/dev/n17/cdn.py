"""n17: cuDNN (frontend) BF16 matmul engines vs F.linear at the proj shapes, cold-L2 graph replays (H100).
python dev/n17/cdn.py 128:qkv_gate 256:pooler ..."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import cudnn, bench, reference
W = reference.generate_weights(reference.WEIGHTS_SEED)
h = cudnn.create_handle()
e = bench.empty_cost()
for a in sys.argv[1:]:
    m, layer = a.split(":"); m = int(m)
    w = W[layer].cuda(); n, k = w.shape
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x, w)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT,
                      compute_data_type=cudnn.data_type.FLOAT, handle=h)
    A = g.tensor(name="A", dim=[1, m, k], stride=[m * k, k, 1], data_type=cudnn.data_type.BFLOAT16)
    B = g.tensor(name="B", dim=[1, k, n], stride=[n * k, 1, k], data_type=cudnn.data_type.BFLOAT16)
    C = g.matmul(name="mm", A=A, B=B, compute_data_type=cudnn.data_type.FLOAT)
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    g.validate(); g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK])
    g.check_support(); g.build_plans(cudnn.build_plan_policy.ALL)
    np_ = g.get_execution_plan_count()
    ws = torch.empty(max(g.get_workspace_size_plan_at_index(i) for i in range(np_)) + 1, device="cuda", dtype=torch.uint8)
    vp = {A: x, B: w.t(), C: out}
    tl = bench.time_graph(lambda: F.linear(x, w, out=None), reps=100) - e
    res = []
    for i in range(np_):
        try:
            def run(i=i):
                cudnn.set_stream(h, torch.cuda.current_stream().cuda_stream)
                g.execute_plan_at_index(vp, ws, i, handle=h)
            run(); torch.cuda.synchronize()
            err = (out.float() - ref.float()).abs().max().item()
            t = bench.time_graph(run, reps=100) - e
            res.append((t, i, g.get_plan_name_at_index(i)[:60], err))
        except Exception as ex:
            res.append((1e9, i, str(ex)[:60], -1))
    res.sort()
    print(f"m{m} {layer}: F.linear {tl:.2f}; cudnn plans {np_}: " + " | ".join(f"{t:.2f} #{i} {nm} err{er:.2g}" for t, i, nm, er in res[:4]), flush=True)
