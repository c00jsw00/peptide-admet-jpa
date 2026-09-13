import subprocess, json, time
def doiorg(doi):
    for a in range(3):
        try:
            r=subprocess.run(['curl','-s','-L','--max-time','30','-H','Accept: application/vnd.citationstyles.csl+json',f'https://doi.org/{doi}'],capture_output=True,text=True,timeout=45)
            d=json.loads(r.stdout)
            return d.get('title','?'), [x.get('family','') for x in d.get('author',[])[:5]], d.get('container-title','?'), (d.get('issued',{}).get('date-parts',[[None]])[0][0])
        except Exception as e:
            if a==2: return None,None,None,str(e)[:60]
            time.sleep(2)
lines=[]
def p(*a): lines.append(' '.join(str(x) for x in a))
# [1] and [2] real DOIs from the manuscript
for lbl,doi in [('[1] Hornsby JCR','10.1016/j.jconrel.2026.114895'),
                ('[2] pepADMET JCIM 5c02518','10.1021/acs.jcim.5c02518'),
                ('[7] cyclic 2512.06971 (confirm broken)','10.48550/arXiv.2512.06971')]:
    t,au,c,y = doiorg(doi)
    p(f'{lbl} -> {y} | {t} | {"; ".join(au or [])} | {c}')
    time.sleep(0.5)
open(r'C:/Users/c00jsw00/openclaw-peptide-admet/_doi4.txt','w',encoding='utf-8').write('\n'.join(lines))
print('DONE')
