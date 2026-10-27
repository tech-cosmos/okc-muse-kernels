import csv,sys,collections
L=[l for l in open(sys.argv[1]) if l.startswith('"')]
rows=list(csv.reader(L)); h=rows[0]; d=collections.OrderedDict()
for r in rows[1:]:
    k=(r[h.index("ID")],r[h.index("Kernel Name")][:36]); d.setdefault(k,{})[r[h.index("Metric Name")]]=(r[h.index("Metric Value")],r[h.index("Metric Unit")])
for (i,n),m in d.items():
    f=lambda x:float(m[x][0].replace(',',''))
    t=f('gpu__time_duration.sum'); ut=m['gpu__time_duration.sum'][1]
    rd=f('dram__bytes_read.sum'); u=m['dram__bytes_read.sum'][1]
    print(f"{i:>3} {n:36s} t={t:8.2f}{ut} rd={rd:8.2f}{u} wr={m['dram__bytes_write.sum'][0]}{m['dram__bytes_write.sum'][1]} pct={m['dram__throughput.avg.pct_of_peak_sustained_elapsed'][0]}")
