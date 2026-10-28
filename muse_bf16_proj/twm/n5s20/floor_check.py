import sys, torch  # proj n5s20: judge floor formula vs generate_input routings
sys.path.insert(0, '/home/daytona/r-moe/muse_moe_w4a16')
import reference as R
PE = 3*4096*4096*(0.5+1/16)
for bs, fl in ((128, 101.3), (256, 108.9)):
    D = []; L = []; RR = []
    for seed in range(60):
        ids = R.generate_input(bs, "ep8", 1000+seed)["topk_ids"]
        c = torch.bincount(ids[ids < 32].long().flatten(), minlength=32).cpu()
        D.append((c > 0).sum().item()); L.append(c.sum().item())
        RR.append(sum(-(-x//64) - 1 for x in c.tolist() if x > 64))
    d = sum(D)/len(D)
    for bw in (3.35e12, 4.0e12, 4.8e12):
        print(bs, "mean distinct %.1f routes %.0f" % (d, sum(L)/len(L)), "bw %.2f floor_pred %.1f" % (bw/1e12, 1e6*(0.75*d*PE + bs*4096*4)/bw), "judge", fl,
              "| full-distinct@4.0 %.1f" % (1e6*d*PE/4e12), "mean extra 64-passes %.2f" % (sum(RR)/len(RR)))
