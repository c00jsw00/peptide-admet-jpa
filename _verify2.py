"""Comprehensive fact verification for manuscript edit. Writes _verify2.txt."""
import json, os, re, subprocess, sys

OUT = open('_verify2.txt', 'w', encoding='utf-8')
def p(*a):
    s = ' '.join(str(x) for x in a)
    OUT.write(s + '\n')

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')

# ---------- environment ----------
p('=== ENV ===')
r = subprocess.run([PY, '-c', 'import sys,torch,sklearn,lightgbm,rdkit;print("py",sys.version.split()[0]);print("torch",torch.__version__);print("sklearn",sklearn.__version__);print("lgbm",lightgbm.__version__);print("rdkit",rdkit.__version__)'], capture_output=True, text=True, cwd=ROOT)
p(r.stdout, r.stderr[:200])
for pkg in ['tabpfn', 'dgl']:
    r = subprocess.run([PY, '-c', f'import {pkg};print("{pkg}",getattr({pkg},"__version__","?"))'], capture_output=True, text=True, cwd=ROOT)
    p(r.stdout.strip(), (r.stderr[:150] if r.returncode else ''))
try:
    r = subprocess.run('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv', shell=True, capture_output=True, text=True, timeout=20)
    p('GPU:', (r.stdout.strip() or r.stderr.strip())[:150])
except Exception as e:
    p('GPU: err', e)

# ---------- KPGT base.pth param count ----------
p('=== KPGT PARAMS ===')
r = subprocess.run([PY, '-c',
    'import torch;sd=torch.load(r"C:/Users/c00jsw00/_kpgt_weights/pretrained/base/base.pth",map_location="cpu");'
    'tot=sum(v.numel() for v in sd.values() if hasattr(v,"numel"));'
    'print("total_params_M",round(tot/1e6,1));'
    'import collections;c=collections.Counter(k.split(".")[0] for k in sd);print(dict(c))'],
    capture_output=True, text=True, cwd=ROOT)
p(r.stdout.strip(), r.stderr[-300:] if r.returncode else '')

# ---------- KPGT finetune constants ----------
p('=== KPGT FINETUNE CONSTANTS ===')
src = open(r'C:/Users/c00jsw00/_kpgt_finetune_gpu.py', encoding='utf-8').read()
for mm in re.finditer(r'^(EPOCHS?|WARMUP|TOTAL|LR|BATCH|PATIENCE|MAX_?\w*|SEEDS?|LR_?\w*|BASE_LR|SCHED)\w*\s*=\s*.+', src, re.M):
    p('  ' + mm.group(0).strip()[:120])
for mm in re.finditer(r'.*(lr=|lr_|weight_decay|patience|warmup|total_steps|epochs|max_epochs)[^#\n]*', src, re.I):
    line = mm.group(0).strip()
    if line and len(line) < 130 and not line.startswith('def '):
        p('  | ' + line)

# ---------- route JSONs (full relevant numbers) ----------
p('=== ROUTE JSONS ===')
def load(n): return json.load(open(os.path.join(ROOT, 'analysis', n), encoding='utf-8'))

d = load('route1_results.json')
p('R1 keys:', list(d.keys()))
p('R1:', json.dumps({k: v for k, v in d.items() if not isinstance(v, (list, dict))}, default=str))
for k in d:
    if isinstance(d[k], dict):
        p(f'  R1.{k}:', json.dumps(d[k], default=str)[:700])

d = load('route2_results.json')
p('R2:', json.dumps(d, default=str)[:900])
d = load('route3_results.json')
p('R3:', json.dumps(d, default=str)[:1200])
d = load('route4_results.json')
p('R4:', json.dumps(d, default=str)[:700])
d = load('route5_results.json')
p('R5:', json.dumps(d, default=str)[:1500])
d = load('label_avg_results.json')
p('R8 label_avg:', json.dumps(d, default=str)[:900])

d = load('chemberta_results.json')
p('R6 chemberta keys:', list(d.keys()))
p('R6 (no seeds):', json.dumps({k: v for k, v in d.items() if k not in ('seeds', 'per_seed', 'all_runs')}, default=str)[:1500])

d = load('route9_kpgt_results.json')
p('R9 KPGT best:', json.dumps(d.get('best'), indent=1, default=str))
p('R9 KPGT summary:', json.dumps(d.get('summary'), indent=1, default=str) if d.get('summary') else '(none)')
for k in d:
    if k not in ('per_seed', 'best', 'summary'):
        p(f'R9.{k}:', json.dumps(d[k], default=str)[:600])

d = load('peptiverse_results.json')
p('PV top:', {k: v for k, v in d.items() if k != 'endpoints'})
for ep, v in d.get('endpoints', {}).items():
    p(f'PV.{ep}:', json.dumps(v, default=str)[:1200])

# ---------- models_v4 summary + caco2 ----------
p('=== MODELS_V4 ===')
try:
    s = json.load(open(os.path.join(ROOT, 'models_v4', 'summary.json'), encoding='utf-8'))
    p('summary:', json.dumps(s, default=str)[:1200])
except Exception as e:
    p('summary err', e)
_cp = os.path.join(ROOT, 'models_v4', 'caco2', 'metrics.json')
m = json.load(open(_cp, encoding='utf-8')) if os.path.exists(_cp) else None
if m:
    p('caco2:', json.dumps({k: m[k] for k in list(m.keys())[:10]}, default=str)[:500])
    p('caco2 splits.primary counts:', json.dumps(m.get('splits', {}).get('primary', {}), default=str)[:300])

# ---------- route script model classes ----------
p('=== ROUTE SCRIPT MODELS ===')
for fn in ['round1_feature_ablation.py', 'round2_rank_gaussian.py', 'round3_strong_2d.py',
           'soft_blend.py', 'label_avg_experiment.py', 'chemberta_retrain.py', 'tobit_censored.py']:
    fp = os.path.join(ROOT, 'analysis', fn)
    if not os.path.exists(fp):
        p(f'{fn}: MISSING'); continue
    t = open(fp, encoding='utf-8').read()
    mods = []
    for pat in [r'import\s+lightgbm', r'from\s+lightgbm\s+import\s+[\w, ]+', r'LGBMRegressor',
                r'import\s+torch', r'nn\.Module', r'MixedADMETMLP', r'XGBRegressor', r'from\s+torch\s+import',
                r'class\s+\w*MLP\w*', r'SklearnMLPRegressor', r'GradientBoosting']:
        for mm in re.finditer(pat, t):
            mods.append(mm.group(0))
    p(f'{fn}: {sorted(set(mods))[:10]}')

# ---------- caco2 ceiling ----------
p('=== CACO2 CEILING ===')
r = subprocess.run(['grep', '-rn', 'ceiling', os.path.join(ROOT, 'analysis') + '/'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if 'caco' in line.lower() or '0.5' in line:
        p('  ' + line[:160])

OUT.close()
print('DONE')
