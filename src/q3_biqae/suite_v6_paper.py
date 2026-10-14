"""Execute the independent paper-equation replication, retaining every trial."""
from dataclasses import replace
from time import perf_counter
import json
import numpy as np
import pandas as pd
from .config import ROOT,write_json,digest
from .amplitude import estimate_payload as original
from .measurement import Measurement
from .paper_biqae import estimate
from .suite_risk import configure,load_inputs,result_row
from .suite_v5_variants import summaries
from .exact_reference import risk
from . import es_estimation,var_search

DEST=ROOT/'artifacts/validation_suite_v6/paper'
METHODS=['paper_iqae_cp','grid_biqae','paper_normal','paper_beta']

def dispatch(p,y,method,config,budget,seed,delta=None,epsilon=None):
    if method not in ('paper_normal','paper_beta','paper_iqae_cp'):
        return original(p,y,method,config,budget,seed,delta,epsilon)
    return estimate(Measurement(p,y,config,seed),config,budget,
        config.failure_probability if delta is None else delta,
        config.epsilon if epsilon is None else epsilon,method,seed+170003)

def main():
    DEST.mkdir(parents=True,exist_ok=True);start=perf_counter()
    cfg=replace(configure(),max_rounds=10000,shots_per_round=256)
    write_json(DEST/'formula_mapping.json',dict(source='https://arxiv.org/html/2507.23074v2',
        author_code='https://github.com/Kirin0570/BIQAE/blob/main/src/biae.py',
        mappings={'Eq4/15':'inverse: odd quadrant reverses endpoints',
        'Eq20 Algorithm1':'normal_update: analytic precision addition with cumulative stage data',
        'Eq21 Algorithm2':'normal quantile clipped to [0,1]',
        'Algorithm3':'normal_prior: delta-method transformed mean and variance',
        'Eq25 Algorithm4':'Beta prior shapes plus cumulative successes/failures',
        'Algorithm5':'scipy Beta equal-tail quantiles then inverse branch',
        'Algorithm6':'1000 independent posterior Beta draws, transformed, scipy maximum likelihood Beta fit; no posterior grid'},
        distinctions=['Paper itself approximates transported Normal/Beta families; not exact Bayesian equivalence.',
        'Normal initial noninformative variance infinity; boundary sample means use Eq20 zero-variance limit and are logged. No pseudo-count repair.',
        'Largest legal standard-schedule power with ratio >=2; paper author code searches with coarser decrement at high powers.',
        'External cap k31 and A-call budget65536 shared; paper native unrestricted scaling theorem not tested.',
        'Paper per-stage alpha spending used; repeated looks and approximate priors do not establish frequentist 95% coverage.',
        'IQAE-CP uses the same cumulative-stage schedule as paper variants; grid retains its original scheduler. Comparison concerns complete algorithms.',
        'Author code only exposes Beta; Normal implemented from paper Algorithms1-3, not claimed byte-for-byte author reproduction.']))
    write_json(DEST/'protocol.json',dict(methods=METHODS,seeds=list(range(100,200)),
        amplitudes=[.01,.025,.05,.3,.5,.9,.99],amplitude_batches=[10,256],amplitude_epsilon=.005,
        risk_alphas=[.95,.975],risk_batch=256,budget=65536,config=cfg.to_dict(),
        formula_map_sha=digest(DEST/'formula_mapping.json'),
        sources={p.name:digest(p) for p in (ROOT/'src/q3_biqae').glob('*paper*.py')},
        model_sha=digest(ROOT/'artifacts/feasibility_trained_v2/models/trained_risk_model.npz'),
        initial_attempt='aborted before first 100-seed group completed: original IQAE discards within-stage observations, unsuitable for paper 10-shot setting; initial protocol retained; no outcomes used for selection'))
    rows=[];logs=[]
    for batch in [10,256]:
        for amp in [.01,.025,.05,.3,.5,.9,.99]:
            for method in METHODS:
                for seed in range(100,200):
                    key=dict(batch=batch,amplitude=amp,method=method,seed=seed);tick=perf_counter()
                    r=dispatch([1-amp,amp],[0.,1.],method,replace(cfg,shots_per_round=batch),65536,seed,epsilon=.005)
                    rows.append(key|dict(status='ok',estimate=r.estimate,error=abs(r.estimate-amp),covered=r.interval[0]<=amp<=r.interval[1],
                        ci_low=r.interval[0],ci_high=r.interval[1],width=r.interval[1]-r.interval[0],A_calls=r.ledger.oracle_A_calls,
                        shots=r.ledger.shots_total,runtime_ms=(perf_counter()-tick)*1000,converged=r.converged,
                        normal_zero_variance=any(x.get('normal_zero_variance',False) for x in r.rounds)))
                    assert r.ledger.oracle_A_calls<=65536
                    logs.append(key|dict(rounds_json=json.dumps(r.rounds,allow_nan=False)))
                print('amplitude',batch,amp,method,flush=True)
                pd.DataFrame(rows).to_csv(DEST/'amplitude_results.csv',index=False)
    adf=pd.DataFrame(rows);pd.DataFrame(logs).to_parquet(DEST/'amplitude_rounds.parquet',index=False)
    summaries(adf,['batch','amplitude','method'],True).to_csv(DEST/'amplitude_summary.csv',index=False)
    summaries(adf,['batch','method'],True).to_csv(DEST/'amplitude_aggregate.csv',index=False)
    es_estimation.estimate_payload=var_search.estimate_payload=dispatch
    dist,raw,weights=load_inputs();rows=[];logs=[]
    for alpha in [.95,.975]:
        truth=risk(dist.losses(weights),dist.probabilities,alpha)
        for method in METHODS:
            for seed in range(100,200):
                key=dict(alpha=alpha,method=method,seed=seed,budget=65536)
                r=es_estimation.estimate_risk(dist,alpha,65536,seed,method,replace(cfg,alpha=alpha),weights)
                rows.append(key|result_row(r,truth));logs.append(key|dict(rounds_json=json.dumps(r['logs'],allow_nan=False)))
            print('risk',alpha,method,flush=True)
            pd.DataFrame(rows).to_csv(DEST/'risk_results.csv',index=False)
    rdf=pd.DataFrame(rows);pd.DataFrame(logs).to_parquet(DEST/'risk_rounds.parquet',index=False)
    summaries(rdf,['alpha','method']).to_csv(DEST/'risk_summary.csv',index=False)
    write_json(DEST/'status.json',dict(status='complete',amplitude_trials=len(adf),risk_trials=len(rdf),
        normal_zero_variance_trials=int(adf.normal_zero_variance.sum()),runtime_s=perf_counter()-start,
        failures=0,grid_dependency=False,replication='paper Algorithms1-6 independently reproduced; external capped evaluation; no exact Bayesian equivalence claim'))
    write_json(DEST/'checksums.json',{p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__': main()
