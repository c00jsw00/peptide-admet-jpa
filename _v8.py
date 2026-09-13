import os, re, json
ROOT = r'C:/Users/c00jsw00/openclaw-peptide-admet'
out = []
def p(*a): out.append(' '.join(str(x) for x in a))

# 1) Which model do classical routes train?
for f in sorted(os.listdir(os.path.join(ROOT,'analysis'))):
    if not f.endswith('.py'): continue
    fp = os.path.join(ROOT,'analysis',f)
    s = open(fp,encoding='utf-8',errors='ignore').read()
    if re.search(r'LightGBM|lgbm', s):
        p(f, '-> LightGBM:', re.findall(r'LightGBM|lgbm', s)[:2])
    if re.search(r'MixedADMETMLP|admet_mlp|load_baseline|baseline_model', s):
        p(f, '-> MLP/baseline:', re.findall(r'MixedADMETMLP|admet_mlp\w*|load_baseline\w*|baseline_model\w*', s)[:4])

# 2) common.py: what feature matrix does it build (desc/morgan/molf dims)?
s = open(os.path.join(ROOT,'analysis','common.py'),encoding='utf-8').read()
p('=== common.py ===')
for line in s.splitlines():
    if re.search(r'217|2048|768|3033|morgan|Morgan|molf|MolFormer|molformer|desc', line):
        p('  ', line.strip()[:150])

# 3) route9 md: KPGT param count / size / GPU / epochs / base.pth
s = open(os.path.join(ROOT,'analysis','route9_tabpfn_kpgt.md'),encoding='utf-8').read()
p('=== route9 md key facts ===')
for line in s.splitlines():
    if re.search(r'M\b|MB|param|V100|GPU|epoch|base\.pth|447|1\.6|frozen|0\.35|42, 123|seed', line, re.I):
        p('  ', line.strip()[:160])

open(os.path.join(ROOT,'_v8.txt'),'w',encoding='utf-8').write('\n'.join(out))
print('DONE', len(out))
