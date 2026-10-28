# proj n5s11: bs128/bs256 twm cost model from H200 item costs (moe NOTES s15 5bbb03ca, s16)
import sys; sys.path.insert(0,'.')
import torch, reference, math
EXP_MB=28.3; G=132; HBM=4.0e6  # MB/s -> us: MB/(TB/s)= us*1e-6... use MB / 4.0 = us*1e-0? 28.3MB/4TB/s=7.08us
def c4(n):  # RT4 item cost by pass tokens (gemm13 ~ gemm2 avg)
    return 36.6 if n<=8 else 41.7 if n<=16 else 45.5 if n<=24 else 49.0
for bs in (128,256):
  for seed in list(range(1,13)):
    d=reference.generate_input(bs=bs,sharding="ep8",seed=seed,device="cuda")
    ids=d["topk_ids"].flatten(); cnt=torch.bincount(ids[ids<32],minlength=32).tolist()
    cnt=[c for c in cnt if c>0]; E=len(cnt)
    items=0; hot=0
    for n in cnt:
      if n>32:
        p=math.ceil(n/64); items+=48*35.5+ (p-1)*48*29; hot+=48*35.5+(p-1)*48*29
      else: items+=24*c4(n)
    stream=E*EXP_MB/4.0
    print(f"bs{bs} s{seed:2d} E{E} max{max(cnt):3d} pairs{sum(cnt)} stream@4.0 {stream:6.1f} items/G {items/G:6.1f} hot/G {hot/G:5.1f} light-at-HBM-share {(items-hot)/G:6.1f}")
