import csv, sys
rows=list(csv.reader(open(sys.argv[1])))[2:]
tot=sum(int(r[37]) for r in rows); alls=sum(int(r[2]) for r in rows)
print("total samples", alls, "long_sb", tot)
idx=sorted(range(len(rows)), key=lambda i:-int(rows[i][37]))[:12]
for i in idx:
    print(f"{i:5d} sb={rows[i][37]:>6} all={rows[i][2]:>6} ex={rows[i][5]:>8} | {rows[i][1].strip()}")
    for j in range(max(0,i-6), i):
        if 'LD' in rows[j][1] or 'SYNCS' in rows[j][1]: print(f"        {j:5d} {rows[j][1].strip()}")
