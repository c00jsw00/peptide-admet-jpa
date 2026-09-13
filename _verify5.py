import json, os, re, subprocess, csv
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = open('_verify5.txt','w',encoding='utf-8')
def p(*a): OUT.write(' '.join(str(x) for x in a)+'\n')

# 1) Grep analysis + repo for GBM / mordred usage
p('=== GBM / mordred usage in analysis & repo ===')
r = subprocess.run(['grep','-rniIl','xgboost\|lightgbm\|lgbm\|mordred', os.path.join(ROOT,'analysis'), os.path.join(ROOT,'train_pepadmet_model.py'), os.path.join(ROOT,'admet_model.py')], capture_output=True, text=True)
p('files matching:', r.stdout.strip() or '(none)')
r2 = subprocess.run(['grep','-rni','mordred', os.path.join(ROOT,'analysis')], capture_output=True, text=True)
p('mordred lines:', (r2.stdout.strip()[:400]) or '(none)')
r3 = subprocess.run(['grep','-rni','xgboost\|lightgbm', os.path.join(ROOT,'analysis')], capture_output=True, text=True)
p('gbm lines:', (r3.stdout.strip()[:300]) or '(none)')

# 2) Route 8 full JSON
p('')
p('=== Route 8 (label_avg) FULL ===')
d = json.load(open(os.path.join(ROOT,'analysis','label_avg_results.json'), encoding='utf-8'))
s = json.dumps(d, indent=1)
# find pampa section + delta
p(s[:2600])

# 3) KPGT first author via CSL
p('')
p('=== KPGT 2206.03364 authors ===')
r = subprocess.run(['curl','-sL','--max-time','25','-H','Accept: application/vnd.citationstyles.csl+json','https://doi.org/10.48550/arXiv.2206.03364'], capture_output=True, text=True, timeout=40)
try:
    d=json.loads(r.stdout); p('title:', d.get('title')); p('authors:', [a.get('family') for a in d.get('author',[])][:8]); p('issued:', d.get('issued',{}).get('date-parts'))
except Exception as e: p('fail', r.stdout[:80])

# 4) Caco-2 sigma/SKIP heavy; clean small compute
p('')
p('=== Caco-2 sigma & SD (clean) ===')
with open(os.path.join(ROOT,'data','pepadmet_caco2.csv'), encoding='utf-8-sig', newline='') as f:
    rd=list(csv.DictReader(f))
ycol=[c for c in rd[0] if 'Caco2' in c or 'Caco-2' in c][0]
ys=[]
for row in rd:
    try: ys.append(float(row[ycol]))
    except: pass
import numpy as np
ys=np.array(ys); F=-10.0; nf=ys>F+1e-6
sd_tot=float(np.std(ys, ddof=0)); sd_nf=float(np.std(ys[nf], ddof=0))
p(f'col={ycol} n={len(ys)} floor={(~nf).sum()} ({100*(~nf).sum()/len(ys):.1f}%)  SD_total={sd_tot:.4f} SD_nonfloor={sd_nf:.4f}  ratio_sd={sd_nf/sd_tot:.4f}  sigma2/V={float(np.var(ys[nf]))/float(np.var(ys)):.4f}')

# 5) PAMPA split counts + censored
p('')
p('=== PAMPA split + censored ===')
with open(os.path.join(ROOT,'data','pepadmet_pampa_mdck.csv'), encoding='utf-8-sig', newline='') as f:
    rd=list(csv.DictReader(f))
ycol=[c for c in rd[0] if 'PAMPA' in c][0]
ys=[]
for row in rd:
    try: ys.append(float(row[ycol]))
    except: pass
ys=np.array(ys)
p(f'col={ycol} total_rows={len(ys)} unique={len(set(ys))}  censored={(ys<=-10+1e-6).sum()} ({100*(ys<=-10+1e-6).sum()/len(ys):.2f}%)')

# 6) peptiverse overlap stat from JSON (no parquet)
p('')
p('=== PeptiVerse overlap (from JSON, no parquet) ===')
try:
    pv = json.load(open(os.path.join(ROOT,'analysis','peptiverse_results.json'), encoding='utf-8'))
    s = json.dumps(pv)
    for key in ['overlap','shared','shared_smiles','max_abs','label_diff','discrepan','n_our','n_pv','our_n','pv_n']:
        for m in re.findall(r'"([^"]*'+key+r'[^"]*)"\s*:\s*([-\d.eE+]+)', s)[:4]:
            p('  ', m)
except Exception as e: p('peptiverse json fail', str(e)[:80])
OUT.close(); print('DONE')
