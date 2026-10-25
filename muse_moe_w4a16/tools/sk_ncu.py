import sys; sys.path.insert(0,'.')
import torch, bench, submission, reference
w, st = bench.state()
bs = int(sys.argv[1]); c = sys.argv[2]
if c.startswith('old'):
    p = int(c[3:]); submission.PLAN_OVERRIDE = None
    submission._plan_saved = submission._plan
    import types
    def k(data, st):  # old path regardless of M
        M = data["hidden_states"].shape[0]
        saved = submission.kernel_fn.__code__
    # emulate old: monkeypatch threshold
    src = open('submission.py').read().replace("if M <= 32:\n        sk =", "if False:\n        sk =")
    ns = {}
    mod = types.ModuleType('sub_old'); mod.__dict__['__file__'] = 'submission.py'
    exec(compile(src, 'submission.py', 'exec'), mod.__dict__)
    mod.PLAN_OVERRIDE = lambda M: (p, p)
    fn = mod.kernel_fn
else:
    if c.startswith("t"): submission.TK2_NSTG = int(c[1:])
    else: submission.SK_CFG = int(c); submission.TK2_NSTG = 0
    fn = submission.kernel_fn
data = reference.generate_input(bs=bs, sharding="ep8", seed=int(sys.argv[3]) if len(sys.argv) > 3 else 7)
fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
