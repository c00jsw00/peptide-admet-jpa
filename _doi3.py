import subprocess, json, time
lines=[]
def p(*a): lines.append(' '.join(str(x) for x in a))

def doiorg(doi):
    for attempt in range(3):
        try:
            r=subprocess.run(['curl','-s','-L','--max-time','30','-H','Accept: application/vnd.citationstyles.csl+json',f'https://doi.org/{doi}'],capture_output=True,text=True,timeout=45)
            d=json.loads(r.stdout)
            return d.get('title','?'), [a.get('family','') for a in d.get('author',[])[:6]], d.get('container-title','?'), (d.get('issued',{}).get('date-parts',[[None]])[0][0])
        except Exception as e:
            if attempt==2: return None, None, None, str(e)[:50]
            time.sleep(2)

def crossref_title(q, rows=5):
    r=subprocess.run(['curl','-s','--max-time','30',f'https://api.crossref.org/works?query.bibliographic={q}&rows={rows}'],capture_output=True,text=True,timeout=45)
    items=json.loads(r.stdout)['message']['items']
    return [(it.get('DOI'), (it.get('title') or ['?'])[0][:70], (it.get('container-title') or ['?'])[0], it.get('issued',{}).get('date-parts',[['?']])[0][0]) for it in items]

# Mordred correct DOI
t,a,c,y = doiorg('10.1186/s13321-018-0258-y')
p(f'Mordred 10.1186/s13321-018-0258-y -> {y} | {t} | {"; ".join(a or [])} | {c}')

# MoLFormer: crossref search
for it in crossref_title('MoLFormer+Pre-training+Molecular+Transformers+Kueck'):
    p('MOLFORMER:', it)
time.sleep(1)
for it in crossref_title('MoLFormer+transformers+molecular+property+prediction'):
    p('MOLFORMER2:', it)

# Cyclic peptide permeability: search
time.sleep(1)
for it in crossref_title('cyclic+peptide+permeability+prediction+benchmarking+platform'):
    p('CYCLICPEPT:', it)
time.sleep(1)
for it in crossref_title('PeptiVerse+unified+platform+therapeutic+peptide'):
    p('PEPTIVERSE:', it)

# KPGT authors + venue check
t,a,c,y = doiorg('10.48550/arXiv.2206.03364')
p(f'KPGT -> {y} | {t} | {"; ".join(a or [])}')

open('_doi3.txt','w',encoding='utf-8').write('\n'.join(lines))
print('DONE')
