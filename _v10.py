import json, os, subprocess, statistics as st
ROOT = r'C:/Users/c00jsw00/openclaw-peptide-admet'
out=[]
def p(*a): out.append(' '.join(str(x) for x in a))

# 1) KPGT mean/sd recomputed (population vs sample)
kp=json.load(open(os.path.join(ROOT,'analysis','route9_kpgt_results.json')))
ps=kp['per_seed']
allv={s: max(v, key=lambda r: r['val_r2']) for s,v in ps.items()}
r2s=[allv[s]['test_r2'] for s in ('42','123','7')]
nfs=[allv[s]['test_nf'] for s in ('42','123','7')]
sp=[allv[s].get('spearman', allv[s].get('spearman_all')) for s in ('42','123','7')]
p(f'KPGT per-seed test_r2={r2s}  mean={st.mean(r2s):.4f} popsd={st.pstdev(r2s):.4f} samplesd={st.stdev(r2s):.4f}')
p(f'KPGT per-seed test_nf={nfs} mean={st.mean(nfs):.4f} popsd={st.pstdev(nfs):.4f} samplesd={st.stdev(nfs):.4f}')
p(f'KPGT per-seed spearman={sp}')
# what keys does per_seed have
p('per_seed[42][0] keys:', list(ps['42'][0].keys()))
# best epoch val_r2
for s in ('42','123','7'):
    b=allv[s]; p(f'  seed {s}: best_epoch={b["epoch"]} val_r2={b["val_r2"]} test_r2={b["test_r2"]} nf={b["test_nf"]} spearman={b.get("spearman","?")}')

# 2) floor SS share + ceiling (from r2_ceiling if output cached; else recompute quickly w/ committed model is heavy -> use json)
# route3 already gave floor_n=47 nonfloor_n=1410. ceiling from peptiverse json + memory.
pv=json.load(open(os.path.join(ROOT,'analysis','peptiverse_results.json')))
p('PV PAMPA oracle_ceiling=', pv['endpoints']['PAMPA_MDCK'].get('oracle_ceiling_r2'))
p('PV Caco2 oracle_ceiling=', pv['endpoints']['Caco2'].get('oracle_ceiling_r2'))

# 3) search for the 0.811 / 0.671 / 0.667 shared-molecule source
import re
hits=[]
for f in os.listdir(ROOT):
    if f.endswith(('.md','.json','.txt','.py')) and not f.startswith('.git'):
        fp=os.path.join(ROOT,f)
        try: s=open(fp,encoding='utf-8',errors='ignore').read()
        except: continue
        for m in re.finditer(r'0\.811|0\.671|0\.667|0\.81\b', s):
            line=s[:m.start()].count('\n')+1
            hits.append((f,line,s.strip().split('\n')[line-1][:120]))
for h in hits[:25]: p('SHARED-HIT', h)
open(os.path.join(ROOT,'_v10.txt'),'w',encoding='utf-8').write('\n'.join(out))
print('DONE')
