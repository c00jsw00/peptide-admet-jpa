"""Verification of manuscript claims against repo sources. Writes _verify_out.txt."""
import json, os, subprocess, re, sys

OUT = open('_verify_out.txt', 'w')
def p(*a):
    s = ' '.join(str(x) for x in a)
    OUT.write(s + '\n')
    print(s)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. baseline model metadata ----------
p('=== 1. BASELINE MODEL (models_v4) ===')
for ep in ['pampa_mdck', 'caco2', 'half_life', 'hemolysis']:
    mp = os.path.join(ROOT, 'models_v4', ep, 'meta.json')
    if os.path.exists(mp):
        m = json.load(open(mp))
        p(f'  {ep}: model={m.get("model")} modality={m.get("modality")} '
          f'n_features={m.get("n_features")} input={m.get("input_dim") or m.get("dim")}')
        for k in m:
            if any(t in k.lower() for t in ['loss', 'opt', 'epoch', 'lr', 'batch', 'hidden', 'head', 'arch', 'molformer', 'emb']):
                p(f'      {k} = {m[k]}')
    else:
        p(f'  {ep}: meta.json MISSING')
    mtp = os.path.join(ROOT, 'models_v4', ep, 'metrics.json')
    if os.path.exists(mtp):
        mt = json.load(open(mtp))
        p(f'  {ep} metrics keys: {list(mt.keys())[:12]}')
        for k in ['r2_test', 'rmse_test', 'mae_test', 'r2_val']:
            if k in mt:
                p(f'      {k} = {mt[k]}')

# ---------- 2. KPGT config actually used ----------
p('=== 2. KPGT CONFIG ===')
cfgpath = r'C:/Users/c00jsw00/_kpgt/src/model_config.py'
src = open(cfgpath, encoding='utf-8').read()
# print the dict of configs
for name in re.findall(r'^(base|small|large|medium|default)\s*=', src, re.M):
    pass
m = re.search(r'(BASE_CONFIG|BASE|base)\s*=\s*\{[^}]*\}', src)
if m:
    p('  base block:\n' + '    ' + m.group(0).replace('\n', '\n    ')[:800])
p('  --- all n_heads mentions ---')
for mm in re.finditer(r'.*n_heads.*', src):
    p('   ' + mm.group(0).strip()[:100])
p('  --- all layer mentions ---')
for mm in re.finditer(r'.*(layers|depth).*', src, re.I):
    p('   ' + mm.group(0).strip()[:100])
# what does the finetune script use?
fp = r'C:/Users/c00jsw00/_kpgt_finetune_gpu.py'
if os.path.exists(fp):
    fsrc = open(fp, encoding='utf-8').read()
    p('  --- finetune script config lines ---')
    for mm in re.finditer(r'.*(config|n_heads|num_layers|layer|d_g|path_length|max_path|heads|param|\.pth|base)[^#\n]*', fsrc, re.I):
        line = mm.group(0).strip()
        if len(line) < 120:
            p('   ' + line)

# ---------- 3. KPGT checkpoint size / params ----------
p('=== 3. KPGT CHECKPOINT ===')
for f in os.listdir(r'C:/Users/c00jsw00/_kpgt_ckpt_gpu'):
    fp2 = os.path.join(r'C:/Users/c00jsw00/_kpgt_ckpt_gpu', f)
    p(f'  ckpt {f}: {os.path.getsize(fp2)/1e6:.1f} MB')
# base pretrained weights
for root, dirs, files in os.walk(r'C:/Users/c00jsw00/_kpgt_weights'):
    for f in files:
        fp3 = os.path.join(root, f)
        p(f'  weight {os.path.relpath(fp3, r"C:/Users/c00jsw00/_kpgt_weights")}: {os.path.getsize(fp3)/1e6:.1f} MB')

# ---------- 4. TabPFN details ----------
p('=== 4. TABPFN ===')
d = json.load(open(os.path.join(ROOT, 'analysis', 'route9_tabpfn_results.json')))
p('  top keys:', list(d.keys()))
for k in d:
    if k not in ('per_seed', 'per_feature'):
        p(f'  {k} = {d[k]}')
p('  per_feature (subset):', json.dumps(d.get('per_feature', {}), default=str)[:600])
# script details
tp = os.path.join(ROOT, 'analysis', 'route9_tabpfn.py')
t = open(tp, encoding='utf-8').read()
for mm in re.finditer(r'.*(seeds|SEED|version|import tabpfn|TabPFN|n_estimators|batch|max_num|500|feature)[^#\n]*', t, re.I):
    line = mm.group(0).strip()
    if line and len(line) < 130:
        p('   ' + line)

# ---------- 5. KPGT route9 JSON full ----------
p('=== 5. ROUTE9 KPGT JSON ===')
d = json.load(open(os.path.join(ROOT, 'analysis', 'route9_kpgt_results.json')))
p(json.dumps(d, indent=1, default=str)[:2500])

OUT.close()
print('DONE -> _verify_out.txt')
