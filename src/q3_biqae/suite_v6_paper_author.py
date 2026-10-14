"""Repeat the frozen paper experiment using authors' exact FindNextK."""
import json
import numpy as np
import pandas as pd
from . import paper_biqae,suite_v6_paper
from .paper_author_schedule import next_power,raw,AUTHOR,SOURCE
from .config import ROOT,write_json,digest

DEST=ROOT/'artifacts/validation_suite_v6/paper_author_schedule'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20260917);mismatches=illegal=0;cases=[]
    for _ in range(10000):
        current=int(rng.integers(32));K=2*current+1;q=int(rng.integers(K))
        endpoints=np.sort(rng.uniform(.001,.999,2));theta=(q+endpoints)*np.pi/2/K
        expected,half=AUTHOR(None,current,q%2==0,(theta/(2*np.pi)).tolist(),2.)
        observed,_=raw(theta,current)
        got,branch=next_power(theta,current,31)
        capped=current if expected>31 else expected
        mismatches+=int(expected!=observed or got!=capped)
        if got>current:
            illegal+=int((2*got+1)*theta[0]<branch*np.pi/2-1e-12 or (2*got+1)*theta[1]>(branch+1)*np.pi/2+1e-12)
        cases.append(dict(current=current,lo=theta[0],hi=theta[1],author=expected,capped=capped,observed=got))
    assert mismatches==illegal==0
    pd.DataFrame(cases).to_csv(DEST/'scheduler_audit_cases.csv',index=False)
    write_json(DEST/'scheduler_audit.json',dict(cases=10000,mismatches=mismatches,illegal_intervals=illegal,
        original_source_sha256=digest(SOURCE),execution='unmodified author AST method',
        cap_policy='If original FindNextK proposes k>31, retain current k; budget truncates shots externally.'))
    paper_biqae.next_power=next_power;suite_v6_paper.DEST=DEST
    def metadata(path,obj):
        if path.name=='formula_mapping.json':
            obj['distinctions'][2]='Unmodified authors FindNextK executed; proposals above external cap31 are rejected and current stage continues.'
        if path.name=='protocol.json':
            obj['initial_attempt']='Fresh 6400-trial run; previous linear-schedule results kept separately.'
            obj['author_scheduler_audit_sha256']=digest(DEST/'scheduler_audit.json')
        write_json(path,obj)
    suite_v6_paper.write_json=metadata
    suite_v6_paper.main()
    rounds=bad=0
    for name in ['amplitude','risk']:
        df=pd.read_csv(DEST/f'{name}_results.csv');logs=pd.read_parquet(DEST/f'{name}_rounds.parquet')
        assert len(df)==len(logs)
        for (_,row),(_,log) in zip(df.iterrows(),logs.iterrows()):
            entries=json.loads(log.rounds_json)
            batches=entries if name=='amplitude' else [x for e in entries for x in e['rounds']]
            bad+=int(sum((2*x['k']+1)*x['shots'] for x in batches)!=row.A_calls);rounds+=len(batches)
    assert bad==0
    write_json(DEST/'audit.json',dict(passed=True,rounds=rounds,ledger_mismatches=bad,
        scheduler_cases=10000,scheduler_mismatches=0,full_author_equivalence=False,
        limitation='Author exact scheduling and paper update modules reproduced; numerical Beta MLE solver differs, Normal is paper-only, external resource caps remain.'))
    write_json(DEST/'checksums.json',{p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':main()
