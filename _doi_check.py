import subprocess, json, time

DOIS = {
    '[1] PepADMET JCR':       '10.1016/j.jconrel.2026.114895',
    '[2] JPA 2025':           '10.1016/j.jpha.2025.101298',
    '[3] PeptiVerse NatCom':  '10.1038/s41467-026-74167-w',
    '[4] MoLFormer JCIM':     '10.1021/acs.jcim.2c00147',
    '[5] KPGT 2206.03364':    '10.48550/arXiv.2206.03364',
    '[6] ChemBERTa 2010.09885':'10.48550/arXiv.2010.09885',
    '[7] CyclicPept 2512.06971':'10.48550/arXiv.2512.06971',
    '[8] AMPBench 2607.25518':'10.48550/arXiv.2607.25518',
    '[9] XGBoost KDD16':      '10.1145/2939672.2939785',
    '[11] Mordred JC12':      '10.1186/s13321-020-00455-y',
    '[12] ECFP JCIM':         '10.1021/ci100050t',
    '[14] DGL 1909.01315':    '10.48550/arXiv.1909.01315',
}
lines = []
def p(*a): lines.append(' '.join(str(x) for x in a))
p('=== DOI RESOLUTION (doi.org CSL) ===')
for label, doi in DOIS.items():
    try:
        r = subprocess.run(['curl','-s','-L','--max-time','25','-H','Accept: application/vnd.citationstyles.csl+json',f'https://doi.org/{doi}'],
                           capture_output=True, text=True, timeout=40)
        d = json.loads(r.stdout)
        title = d.get('title','?')
        authors = [a.get('given','')+' '+a.get('family','') for a in d.get('author',[])[:4]]
        cont = d.get('container-title','?')
        year = (d.get('issued',{}).get('date-parts',[[None]])[0][0])
        p(f'{label}: OK  | {year} | {title} | {"; ".join(authors)} | {cont}')
    except Exception as e:
        p(f'{label}: FAIL {doi} -> {str(e)[:60]}')
    time.sleep(0.3)
open('_doi_check.txt','w',encoding='utf-8').write('\n'.join(lines))
print('DONE')
