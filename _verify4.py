import json, os, re, subprocess

OUT = open('_verify4.txt', 'w', encoding='utf-8')
def p(*a): OUT.write(' '.join(str(x) for x in a) + '\n')
ROOT = os.path.dirname(os.path.abspath(__file__))
def lo(n): return json.load(open(os.path.join(ROOT, n), encoding='utf-8'))

# ---- 1) arXiv DOIs via doi.org (authoritative) ----
arxiv = {
 'R5_KPGT_2206.03364':'10.48550/arXiv.2206.03364',
 'R6_ChemBERTa_2010.09885':'10.48550/arXiv.2010.09885',
 'R10_LightGBM_1706.06067':'10.48550/arXiv.1706.06067',
 'R14_DGL_1909.01315':'10.48550/arXiv.1909.01315',
 'Mordred_13321':'10.1186/s13321-020-00455-y',
 'TabPFNV1_2409.07270':'10.48550/arXiv.2409.07270',
 'ESMC_2401.04735':'10.48550/arXiv.2401.04735',
 'MolFormer_2111.00964':'10.48550/arXiv.2111.00964',
 'TabPFNV2_Nature':'10.1038/s41586-024-08328-6',
}
p('=== arXiv / other DOIs via doi.org ===')
for k,doi in arxiv.items():
    r = subprocess.run(['curl','-sL','--max-time','25','-H','Accept: application/vnd.citationstyles.csl+json',f'https://doi.org/{doi}'], capture_output=True, text=True, timeout=40)
    try:
        d = json.loads(r.stdout)
        p(f'{k:28s} -> {str(d.get("title"))[:58]} | {d.get("container-title","")[:26]} | {d.get("issued",{}).get("date-parts",["?"])[0]}')
    except Exception:
        p(f'{k:28s} -> FAIL: {r.stdout[:60]}')

# ---- 2) full non-floor values from route JSONs ----
p('')
p('=== non-floor R2 per route (full JSON) ===')
for n in ['route1_results.json','route2_results.json','route3_results.json','route5_results.json','label_avg_results.json']:
    d = lo('analysis/'+n)
    s = json.dumps(d)
    m = re.findall(r'"([^"]*(?:nonfloor|non_floor|nonfloor_R2|uncensored|nf_?r2|r2_?nf)[^"]*)"\s*:\s*([\d.eE+-]+)', s)
    p(f'-- {n}:')
    for k,v in m[:12]:
        p(f'   {k} = {v}')

# ---- 3) Caco-2 sigma_hat^2/V + ceiling ----
p('')
p('=== Caco-2 ceiling detail ===')
import csv
with open(os.path.join(ROOT,'data','pepadmet_caco2.csv'), encoding='utf-8-sig', newline='') as f:
    rd = list(csv.DictReader(f))
ycol = [c for c in rd[0] if 'Caco2' in c or 'Caco-2' in c][0]
ys=[]
for r in rd:
    try: ys.append(float(r[ycol]))
    except: pass
import numpy as np
ys=np.array(ys); F=-10.0
nf=ys> F+1e-6
V_tot = float(np.var(ys, ddof=0))
sigma2 = float(np.var(ys[nf], ddof=0))
V_nf = V_tot - (float(np.mean(ys[~nf]))**2)*(len(ys[~nf])/len(ys))  # not exact; use var decomposition below
# exact: var(y)=E[y^2]-E[y]^2; censored rows all = F
E = float(np.mean(ys)); E2 = float(np.mean(ys**2))
# non-floor contribution to variance is bounded by sigma2 (within-group) ; ceiling = var(y within non-floor subset normalized)
ceiling = sigma2 / V_tot
p(f'n={len(ys)} floor={(~nf).sum()} ({100*(~nf).sum()/len(ys):.1f}%)  V_tot={V_tot:.4f} sigma2_nf={sigma2:.4f}  sigma2/V={sigma2/V_tot:.4f} (ceiling if non-floor R2=1)')

# ---- 4) PeptiVerse overlap ----
p('')
p('=== PeptiVerse overlap (our set vs PV) ===')
import pandas as pd
ours = pd.read_csv(os.path.join(ROOT,'data','pepadmet_pampa_mdck.csv'))
ov = None
cands = list(os.path.join(ROOT,'data','peptiverse',x) for x in os.listdir(os.path.join(ROOT,'data','peptiverse'))) if os.path.isdir(os.path.join(ROOT,'data','peptiverse')) else []
p('peptiverse data dir files:', [os.path.basename(c) for c in cands][:5])
pvf = [c for c in cands if c.endswith(('.csv','.parquet'))]
if pvf:
    try:
        pv = pd.read_csv(pvf[0]) if pvf[0].endswith('.csv') else pd.read_parquet(pvf[0])
        p('PV cols:', list(pv.columns)[:10], 'rows', len(pv))
        scol = [c for c in pv.columns if 'sequence' in c.lower() or 'smiles' in c.lower()]
        p('PV seq col:', scol[:3])
    except Exception as e:
        p('PV load fail:', str(e)[:80])

OUT.close()
print('DONE')
