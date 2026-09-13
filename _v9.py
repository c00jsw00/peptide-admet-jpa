import json, os, sys, subprocess
ROOT = r'C:/Users/c00jsw00/openclaw-peptide-admet'
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
out = []
def p(*a): out.append(' '.join(str(x) for x in a))

# 1) KPGT param count via state dict of base.pth
try:
    import torch
    ck = torch.load(r'C:/Users/c00jsw00/_kpgt_weights/pretrained/base/base.pth',
                    map_location='cpu', weights_only=True)
    sd = ck.get('state_dict', ck)
    total = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
    p(f'KPGT base.pth total params = {total:,}')
    # backbone vs head
    import re
    groups = {}
    for k, v in sd.items():
        if not hasattr(v, 'numel'): continue
        m = re.match(r'(mol_T_layers\.\d+|dist_attn_layer|path_attn_layer|trip_fortrans|atom_emb|bond_emb|trip_emb|head\w*|proj\w*)', k)
        key = m.group(1).split('.')[0] if m else 'other'
        groups[key] = groups.get(key, 0) + v.numel()
    for k in sorted(groups, key=lambda x: -groups[x]):
        p(f'   {k}: {groups[k]:,}')
except Exception as e:
    p('KPGT param FAIL:', repr(e)[:120])

# 2) package versions
import importlib.metadata as imd
for pkg in ['torch', 'lightgbm', 'tabpfn', 'rdkit', 'dgl', 'scikit-learn', 'numpy', 'pandas']:
    try: p(f'ver {pkg} = {imd.version(pkg)}')
    except Exception: p(f'ver {pkg} = N/A')
import sys as _s; p('python', _s.version.split()[0])

open(os.path.join(ROOT, '_v9.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('DONE')
