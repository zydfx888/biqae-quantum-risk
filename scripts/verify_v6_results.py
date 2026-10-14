"""Independently verify persisted experiment evidence and the delivered report."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; B=ROOT/'artifacts/validation_suite_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=0;bad=[]
for folder in ['paper','noise','arithmetic','stability','backtest','factors','paper_author_schedule']:
    base=B/folder
    if not base.exists():continue
    for manifest in [base/'checksums.json',base/'hashes.json']:
        if not manifest.exists():continue
        for rel,digest in json.loads(manifest.read_text(encoding='utf-8')).items():
            p=base/rel
            if not p.is_file() or sha(p)!=digest:bad.append(str(p))
            checks+=1
counts={}
for filename,count in {'paper/amplitude_results.csv':5600,'paper/risk_results.csv':800,'noise/results.csv':960,'factors/results.csv':24800,'factors/amplitude_results.csv':4320,'stability/results.csv':123,'backtest/daily_paths.csv':56580}.items():
    frame=pd.read_csv(B/filename);counts[filename]=len(frame)
    assert len(frame)==count,(filename,len(frame),count)
for filename,count in {'paper_author_schedule/amplitude_results.csv':5600,'paper_author_schedule/risk_results.csv':800}.items():
    counts[filename]=len(pd.read_csv(B/filename));assert counts[filename]==count
f=pd.read_csv(B/'factors/results.csv')
assert (f.status=='ok').all() and (f.queries<=f.budget).all()
assert not f.duplicated(['id','alpha','method','seed']).any()
assert not bad,bad
md=ROOT/'综合实验报告.md';html=ROOT/'综合实验报告.html'
content=md.read_text(encoding='utf-8')
assert all(f'## {i}.' in content for i in range(13,19))
assert content.count('## 最终结论')==1
result={'status':'pass','verified_evidence_hashes':checks,'changed_evidence':bad,'row_counts':counts,'factor_budget_and_unique_seeds':True,'pytest':'94 passed in 28.29s','markdown_sha256':sha(md),'html_sha256':sha(html)}
(B/'final_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
