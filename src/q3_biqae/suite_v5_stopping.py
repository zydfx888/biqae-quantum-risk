"""Paired stopping-rule checks at three common internal accuracy targets."""
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
import json
import numpy as np
import pandas as pd
from .config import ROOT,write_json,digest
from .suite_risk import load_inputs,configure,result_row
from .suite_v4_variants import aggregate
from .exact_reference import risk
from .es_estimation import estimate_risk

OUT=ROOT/'artifacts/validation_suite_v5/stopping'
METHODS=['iqae','grid_biqae','normal_biqae','beta_biqae']

def run():
    start=perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    cfg=replace(configure(),max_rounds=256)
    write_json(OUT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'methods':METHODS,'alphas':[.95,.975],'seeds':list(range(100,200)),
        'epsilon_bp':[2,5,10],'stopping':['interval','budget'],'budget':65536,
        'common_config':cfg.to_dict(),'factor_isolation':'each epsilon comparison shares 256 rounds, prior/grid/budget/seed; changes stopping only',
        'fixed_budget':'disables precision early stop; retains shared safety limits and resolved VaR search termination; unused total budget allowed',
        'reason':'previous 2bp-only stopping pair never triggered precision stopping; additional common5/10bp targets inspect when the rule matters',
        'all_trials_new':True,'model_sha256':digest(ROOT/'artifacts/feasibility_trained_v2/models/trained_risk_model.npz'),
        'source_sha256':digest(__file__)})
    table,_,weights=load_inputs();rows=[];logs=[]
    for eps in [2,5,10]:
        for rule in ['interval','budget']:
            conf=replace(cfg,epsilon=eps/1e4,stopping_rule=rule)
            for alpha in [.95,.975]:
                truth=risk(table.losses(weights),table.probabilities,alpha)
                for method in METHODS:
                    for seed in range(100,200):
                        key={'epsilon_bp':eps,'stopping':rule,'alpha':alpha,'method':method,'seed':seed}
                        result=estimate_risk(table,alpha,65536,seed,method,replace(conf,alpha=alpha),weights)
                        row={**key,**result_row(result,truth),'unused_budget':65536-result['resources']['oracle_A_calls'],'converged':result['converged']}
                        count=sum(q['shots']*(2*q['k']+1) for stage in result['logs'] for q in stage['rounds'])
                        assert count==row['A_calls'] and count<=65536
                        rows.append(row);logs.append({**key,'logs_json':json.dumps(result['logs'])})
                    pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
                    print(eps,rule,alpha,method,flush=True)
    frame=pd.DataFrame(rows)
    pd.DataFrame(logs).to_parquet(OUT/'rounds.parquet',index=False,compression='zstd')
    summary=aggregate(frame,['epsilon_bp','stopping','alpha','method'])
    summary.to_csv(OUT/'summary.csv',index=False)
    pairs=[]
    for (eps,alpha,method),group in frame.groupby(['epsilon_bp','alpha','method']):
        a=group[group.stopping=='interval'].sort_values('seed').reset_index(drop=True)
        b=group[group.stopping=='budget'].sort_values('seed').reset_index(drop=True)
        assert a.seed.equals(b.seed) and len(a)==100
        pairs.append({'epsilon_bp':eps,'alpha':alpha,'method':method,'pairs':100,
            'fewer_query_pairs':int((a.A_calls<b.A_calls).sum()),'equal_query_pairs':int((a.A_calls==b.A_calls).sum()),
            'interval_minus_budget_queries':float((a.A_calls-b.A_calls).mean()),
            'interval_minus_budget_es_abs_error_bp':float((a.es_error_bp-b.es_error_bp).mean()),
            'max_point_difference_bp':float(np.abs(a.es-b.es).max()*1e4),
            'interval_converged_rate':float(a.converged.mean()),'budget_converged_rate':float(b.converged.mean())})
    pd.DataFrame(pairs).to_csv(OUT/'paired.csv',index=False)
    write_json(OUT/'completion.json',{'status':'complete','trials':len(frame),'paired_trials':len(frame)//2,
        'failed_trials':0,'all_budgets_reconstructed':True,'runtime_s':perf_counter()-start})
    write_json(OUT/'checksums.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':run()
