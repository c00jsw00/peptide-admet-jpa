import json, os, re, subprocess

OUT = open('_verify3.txt', 'w', encoding='utf-8')
def p(*a):
    OUT.write(' '.join(str(x) for x in a) + '\n')
ROOT = os.path.dirname(os.path.abspath(__file__))
def load(n): return json.load(open(os.path.join(ROOT, 'analysis', n), encoding='utf-8'))

# non-floor R2 for each route (full walk for *_nonfloor* / *nonfloor* keys)
def walk(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            np_ = path + '.' + str(k)
            if any(s in k.lower() for s in ['nonfloor', 'non_floor', 'uncensored', 'r2_nf', 'nf_']):
                if isinstance(v, (int, float)):
                    p(f'  {np_} = {v}')
            walk(v, np_)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, path + f'[{i}]')

for name in ['route1_results.json','route2_results.json','route3_results.json','route4_results.json',
             'route5_results.json','label_avg_results.json','chemberta_results.json']:
    p('=== ' + name + ' non-floor keys ===')
    walk(load(name))

# route3 two-stage D
d3 = load('route3_results.json')
p('R3 D_two_stage:', json.dumps(d3.get('results',{}).get('D_two_stage_floor_classifier'), default=str)[:600])
# route8 non-floor?
d8 = load('label_avg_results.json')
p('R8 pampa keys:', list(d8.get('pampa',{}).keys()))
p('R8 pampa B_label_avg:', json.dumps(d8.get('pampa',{}).get('B_label_avg'), default=str)[:400])

# route4 best_overall
d4 = load('route4_results.json')
p('R4 best_overall:', json.dumps(d4.get('results',{}).get('best_overall'), default=str)[:300])

# Caco-2 ceiling: run r2_ceiling on caco2 if it supports arg, else compute inline
p('=== CACO2 CEILING (inline) ===')
import numpy as np, pandas as pd
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('common', os.path.join(ROOT,'analysis','common.py'))
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    # find caco2 data
    cand = [os.path.join(ROOT,'data','pepadmet_caco2.csv'), os.path.join(ROOT,'data','pepadmet_caco2_mdck.csv')]
    cpath = next((c for c in cand if os.path.exists(c)), None)
    p('caco2 csv:', cpath)
    if cpath:
        df = pd.read_csv(cpath)
        ycol = [c for c in df.columns if 'caco' in c.lower() or 'mdck' in c.lower() or c.lower()=='logpapp' or 'papp' in c.lower()]
        p('ycol cands:', ycol[:8])
        y = df[ycol[0]].values.astype(float)
        F = -10.0
        floor = y <= F + 1e-6
        ss_total = ((y - y.mean())**2).sum()
        ss_floor = ss_total * (floor.mean())  # approx
        # variance decomposition like r2_ceiling
        y_nf = y[~floor]
        var_nf = y_nf.var()
        var_total = y.var()
        frac_nf = var_nf / var_total
        p(f'caco2 n={len(y)} floor={floor.sum()} ({100*floor.mean():.1f}%)')
        p(f'var_nf/var_total = {frac_nf:.4f} -> ceiling if perfect nonfloor R2=1: {frac_nf:.4f}')
except Exception as e:
    import traceback; p('ERR', e); p(traceback.format_exc()[-500:])

# DOI verification (title/venue/year/authors) via Crossref
def doi(d):
    r = subprocess.run(['curl','-s','--max-time','25',f'https://api.crossref.org/works/{d}'], capture_output=True, text=True, timeout=40)
    try:
        j = json.loads(r.stdout)['message']
        return {'t': (j.get('title') or ['?'])[0][:95], 'v': (j.get('container-title') or ['?'])[0][:40],
                'y': (j.get('issued',{}).get('date-parts') or [[None]])[0][0],
                'au': [a.get('family') for a in j.get('author',[])][:6]}
    except Exception:
        return {'err': r.stdout[:80] or r.stderr[:80]}

p('=== DOIs ===')
refs = {
 '1': '10.1016/j.jconrel.2026.114895',
 '2': '10.1021/acs.jcim.5c02518',
 '3': '10.1038/s41467-026-74167-w',
 '4_tabpfn_v2': '10.1038/s41586-024-08328-6',
 '5_kpgt': '10.48550/arXiv.2206.03364',
 '6_chemberta': '10.48550/arXiv.2010.09885',
 '9_xgboost': '10.1145/2939672.2939785',
 '10_lightgbm': '10.48550/arXiv.1706.06067',
 '11_mordred': '10.1186/s13321-020-00455-y',
 '12_morgan': '10.1021/ci100050t',
 '14_dgl': '10.48550/arXiv.1909.01315',
}
for k, d in refs.items():
    p(k, d, doi(d))

# find ESM-2 reference via crossref search
p('=== ESM-2 search ===')
r = subprocess.run(['curl','-s','--max-time','30','https://api.crossref.org/works?query.bibliographic=ESM-2+evolutionary+scale+model+Mirdita+Science&rows=3'], capture_output=True, text=True, timeout=45)
try:
    for it in json.loads(r.stdout)['message']['items'][:3]:
        p('  ', it.get('DOI'), (it.get('title') or ['?'])[0][:80], (it.get('container-title') or ['?'])[0][:30], it.get('issued',{}).get('date-parts'))
except Exception as e:
    p('  err', e)

# verify v4.2 protocol doc exists in repo
p('=== v4.2 protocol doc ===')
for root2, dirs, files in os.walk(ROOT):
    if '.git' in root2: continue
    for f in files:
        if f.lower() in ('protocol.md','v42.md','protocol_v42.md','v4.2.md'):
            p('  found', os.path.join(root2, f))
OUT.close(); print('DONE')
