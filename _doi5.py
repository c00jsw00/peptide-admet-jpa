import subprocess, json, time
lines=[]
def p(*a): lines.append(' '.join(str(x) for x in a))
def cr(q, rows=6):
    r=subprocess.run(['curl','-s','--max-time','30',f'https://api.crossref.org/works?query.bibliographic={q}&rows={rows}'],capture_output=True,text=True,timeout=45)
    for it in json.loads(r.stdout)['message']['items'][:rows]:
        p(f'   {it.get("DOI")} | {it.get("issued",{}).get("date-parts",[["?"]])[0][0]} | {(it.get("title") or ["?"])[0][:65]} | {(it.get("container-title") or ["?"])[0] if it.get("container-title") else "?"}')
p('--- pepADMET platform search ---')
cr('pepADMET+platform+peptide+ADMET+prediction')
time.sleep(1)
p('--- pepADMET J Chem Inf Model ---')
cr('pepADMET+A+Novel+Computational+Platform+for+Peptide+ADMET+Prediction')
time.sleep(1)
p('--- cyclic peptide permeability benchmark platform arXiv ---')
cr('cyclic+peptide+permeability+prediction+benchmarking+platform')
open(r'C:/Users/c00jsw00/openclaw-peptide-admet/_doi5.txt','w',encoding='utf-8').write('\n'.join(lines))
print('DONE')
