import subprocess, json, time
DOIS = {
    '[4] MoLFormer JCIM 2c00147': '10.1021/acs.jcim.2c00147',
    '[4b] MoLFormer alt 2c00090':'10.1021/acs.jcim.2c00090',
    '[11] Mordred 10.1186/s13321-020-00455-y':'10.1186/s13321-020-00455-y',
    '[11b] Mordred alt 10.1186/s13321-020-00455-y':'10.1186/s13321-020-00455-y',
}
lines=[]
def p(*a): lines.append(' '.join(str(x) for x in a))
p('=== RETRY DOIs ===')
for label, doi in DOIS.items():
    for attempt in range(3):
        try:
            r = subprocess.run(['curl','-s','-L','--max-time','30','-H','Accept: application/vnd.citationstyles.csl+json',f'https://doi.org/{doi}'],
                               capture_output=True, text=True, timeout=45)
            d = json.loads(r.stdout)
            title=d.get('title','?'); authors=[a.get('given','')+' '+a.get('family','') for a in d.get('author',[])[:4]]
            cont=d.get('container-title','?'); year=(d.get('issued',{}).get('date-parts',[[None]])[0][0])
            p(f'{label}: OK | {year} | {title} | {"; ".join(authors)} | {cont}')
            break
        except Exception as e:
            if attempt==2: p(f'{label}: FAIL {doi} -> {str(e)[:50]}')
            time.sleep(1.5)
    time.sleep(0.5)
# Also search crossref for MoLFormer and Mordred by title
for q,label in [('MoLFormer+pre-training+molecular+transformers','MOLFORMER'), ('Mordred+molecular+descriptor+calculator','MORDRED')]:
    try:
        r=subprocess.run(['curl','-s','--max-time','30',f'https://api.crossref.org/works?query.bibliographic={q}&rows=3'],capture_output=True,text=True,timeout=45)
        items=json.loads(r.stdout)['message']['items']
        p(f'--- {label} crossref ---')
        for it in items[:3]:
            p(f'   {it.get("DOI")} | {it.get("title",["?"])[0][:60]} | {it.get("container-title",["?"])[0] if it.get("container-title") else "?"}')
    except Exception as e:
        p(f'{label} crossref FAIL {str(e)[:50]}')
open('_doi2.txt','w',encoding='utf-8').write('\n'.join(lines))
print('DONE')
