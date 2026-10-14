"""Frozen, paired 100-seed comparison of existing posterior implementations."""
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
import json
import numpy as np
import pandas as pd
from .config import ROOT, digest, write_json
from .suite_risk import load_inputs, configure, result_row, cp
from .exact_reference import risk
from .es_estimation import estimate_risk
from .amplitude import estimate_payload

METHODS = ['iqae', 'grid_biqae', 'normal_biqae', 'beta_biqae']
DEST = ROOT/'artifacts/validation_suite_v5/variants'

def summaries(df, keys, amp=False):
    rows=[]
    for key,g in df.groupby(keys):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(keys,key)); n=len(g)
        row.update(n=n, failures=int((g.status!='ok').sum()), mean_A_calls=float(g.A_calls.mean()),
                   mean_shots=float(g.shots.mean()), mean_runtime_ms=float(g.runtime_ms.mean()))
        if amp:
            metrics={'coverage':g.covered, 'success':g.error<=.005}
            row.update(mae=float(g.error.mean()), mean_width=float(g.width.mean()))
        else:
            metrics={'var_coverage':g.var_covered,'es_coverage':g.es_covered,
                     'success_5bp':(g.var_error_bp<=5)&(g.es_error_bp<=5),
                     'success_10bp':(g.var_error_bp<=10)&(g.es_error_bp<=10)}
            row.update(var_mae_bp=float(g.var_error_bp.mean()),es_mae_bp=float(g.es_error_bp.mean()),
                       mean_var_width_bp=float(g.var_width_bp.mean()),mean_es_width_bp=float(g.es_width_bp.mean()))
        for name,values in metrics.items():
            good=values.fillna(False)&(g.status=='ok'); h=int(good.sum()); lo,hi=cp(h,n)
            row.update({name:h/n,name+'_cp_low':lo,name+'_cp_high':hi})
        rows.append(row)
    return pd.DataFrame(rows)

def paired(df, keys, metrics):
    rows=[]; rng=np.random.default_rng(20260916)
    for key,g in df.groupby(keys):
        key=key if isinstance(key,tuple) else (key,)
        reference=g[g.method=='grid_biqae'].set_index('seed')
        for method in ['iqae','normal_biqae','beta_biqae']:
            candidate=g[g.method==method].set_index('seed')
            for metric in metrics:
                difference=(candidate[metric]-reference[metric]).dropna().to_numpy()
                boot=difference[rng.integers(0,len(difference),(10000,len(difference)))].mean(axis=1)
                lo,hi=np.quantile(boot,[.025,.975])
                rows.append(dict(zip(keys,key))|dict(method=method,metric=metric,n=len(difference),
                    mean_difference=float(difference.mean()),bootstrap_low=float(lo),bootstrap_high=float(hi)))
    return pd.DataFrame(rows)

def main():
    started=perf_counter(); cfg=configure()
    DEST.mkdir(parents=True,exist_ok=False)
    write_json(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        methods=METHODS,seeds=list(range(100,200)),risk_alphas=[.95,.975],risk_budgets=[4096,16384,65536],
        amplitudes=[.01,.025,.05,.3,.5,.9,.99],amplitude_budget=65536,amplitude_epsilon=.005,
        config=cfg.to_dict(),reuse=False,all_trials='fresh reruns; no selection',
        limits='Normal/Beta are existing grid-dependent moment projections, not paper-exact algorithms; credible intervals are not guaranteed confidence intervals',
        paired='matched seed; 10000 resamples, marginal descriptive bootstrap CI, no significance or multiple-comparison superiority claim',
        runtime='end-to-end analytic-shot estimator wall time including classical posterior processing; not hardware time; concurrent system load may affect measurements',
        source_hashes={p.name:digest(p) for p in (ROOT/'src/q3_biqae').glob('*.py')},
        model_sha256=digest(ROOT/'artifacts/feasibility_trained_v2/models/trained_risk_model.npz'),
        scenarios_sha256=digest(ROOT/'artifacts/feasibility_trained_v2/generated_scenarios.npz'),
        weights_sha256=digest(ROOT/'artifacts/feasibility_trained_v2/models/policy_lp.json')))
    dist,raw,weights=load_inputs(); rows=[];logs=[]
    for alpha in [.95,.975]:
        truth=risk(dist.losses(weights),dist.probabilities,alpha)
        for budget in [4096,16384,65536]:
            for method in METHODS:
                for seed in range(100,200):
                    key=dict(alpha=alpha,budget=budget,method=method,seed=seed)
                    r=estimate_risk(dist,alpha,budget,seed,method,replace(cfg,alpha=alpha),weights)
                    row=key|result_row(r,truth); assert row['A_calls']<=budget
                    rows.append(row);logs.append(key|dict(rounds_json=json.dumps(r['logs'],allow_nan=False)))
                print('risk',alpha,budget,method,flush=True)
                pd.DataFrame(rows).to_csv(DEST/'risk_results.csv',index=False)
    rdf=pd.DataFrame(rows);pd.DataFrame(logs).to_parquet(DEST/'risk_rounds.parquet',index=False)
    summaries(rdf,['alpha','budget','method']).to_csv(DEST/'risk_summary.csv',index=False)
    paired(rdf,['alpha','budget'],['var_error_bp','es_error_bp','A_calls']).to_csv(DEST/'risk_paired.csv',index=False)
    rows=[];logs=[]
    for amp in [.01,.025,.05,.3,.5,.9,.99]:
        for method in METHODS:
            for seed in range(100,200):
                key=dict(amplitude=amp,method=method,seed=seed);t=perf_counter()
                r=estimate_payload([1-amp,amp],[0.,1.],method,cfg,65536,seed,epsilon=.005)
                assert r.ledger.oracle_A_calls<=65536
                rows.append(key|dict(status='ok',estimate=r.estimate,error=abs(r.estimate-amp),
                    covered=r.interval[0]<=amp<=r.interval[1],ci_low=r.interval[0],ci_high=r.interval[1],
                    width=r.interval[1]-r.interval[0],A_calls=r.ledger.oracle_A_calls,shots=r.ledger.shots_total,
                    runtime_ms=(perf_counter()-t)*1000,converged=r.converged,interval_kind=r.interval_kind))
                logs.append(key|dict(rounds_json=json.dumps(r.rounds,allow_nan=False)))
            print('amplitude',amp,method,flush=True)
            pd.DataFrame(rows).to_csv(DEST/'amplitude_results.csv',index=False)
    adf=pd.DataFrame(rows);pd.DataFrame(logs).to_parquet(DEST/'amplitude_rounds.parquet',index=False)
    summaries(adf,['amplitude','method'],True).to_csv(DEST/'amplitude_summary.csv',index=False)
    summaries(adf,['method'],True).to_csv(DEST/'amplitude_aggregate.csv',index=False)
    paired(adf,['amplitude'],['error','A_calls']).to_csv(DEST/'amplitude_paired.csv',index=False)
    write_json(DEST/'status.json',dict(status='complete',runtime_s=perf_counter()-started,risk_trials=len(rdf),
        amplitude_trials=len(adf),failures=0,duplicate_risk_keys=int(rdf.duplicated(['alpha','budget','method','seed']).sum()),
        duplicate_amplitude_keys=int(adf.duplicated(['amplitude','method','seed']).sum())))
    write_json(DEST/'checksums.json',{p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':
    main()
