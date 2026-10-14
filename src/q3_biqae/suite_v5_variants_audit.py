"""Independent ledger and posterior projection checks; timed diagnostic reruns."""
import json
from time import perf_counter
from datetime import datetime, timezone
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import beta
from .suite_v5_variants import DEST, METHODS
from .suite_risk import load_inputs, configure
from .config import write_json, digest
from .amplitude import estimate_payload
from .es_estimation import estimate_risk

def main():
    cfg=configure(); checks={};projection=[]
    for task in ['amplitude','risk']:
        df=pd.read_csv(DEST/f'{task}_results.csv')
        logs=pd.read_parquet(DEST/f'{task}_rounds.parquet')
        keys=['amplitude','method','seed'] if task=='amplitude' else ['alpha','budget','method','seed']
        joined=df.merge(logs,on=keys,validate='one_to_one'); max_error=0;max_shots_error=0
        for row in joined.to_dict('records'):
            recorded=json.loads(row['rounds_json'])
            rounds=recorded if task=='amplitude' else [r for s in recorded for r in s['rounds']]
            max_error=max(max_error,abs(sum(r['shots']*(2*r['k']+1) for r in rounds)-row['A_calls']))
            max_shots_error=max(max_shots_error,abs(sum(r['shots'] for r in rounds)-row['shots']))
            if task=='amplitude' and row['method']!='iqae':
                theta=(np.arange(cfg.grid_size)+.5)/cfg.grid_size*np.pi/2;a=np.sin(theta)**2
                logp=beta.logpdf(a,*cfg.prior)+np.log(np.sin(2*theta))
                for r in rounds:
                    p=np.clip(np.sin((2*r['k']+1)*theta)**2,1e-15,1-1e-15)
                    logp+=r['successes']*np.log(p)+(r['shots']-r['successes'])*np.log1p(-p)
                posterior=np.exp(logp-logsumexp(logp));exact=float(posterior@a)
                projection.append({k:row[k] for k in keys}|dict(full_grid_replay_estimate=exact,
                    projected_estimate=row['estimate'],projection_difference=row['estimate']-exact,
                    full_grid_replay_error=abs(exact-row['amplitude']),projected_error=row['error']))
        checks[task]=dict(rows=len(joined),max_A_ledger_discrepancy=max_error,max_shots_discrepancy=max_shots_error)
        assert max_error==0 and max_shots_error==0
    pdf=pd.DataFrame(projection);pdf.to_csv(DEST/'posterior_projection_replay.csv',index=False)
    ps=pdf.assign(abs_projection=lambda x:x.projection_difference.abs()).groupby('method').agg(
        mean_abs_projection=('abs_projection','mean'),max_abs_projection=('abs_projection','max'),
        replay_mae=('full_grid_replay_error','mean'),projected_mae=('projected_error','mean'))
    ps.to_csv(DEST/'posterior_projection_summary.csv')
    assert ps.loc['grid_biqae','max_abs_projection']<1e-12
    write_json(DEST/'timing_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        methods=METHODS,seeds=list(range(100,110)),amplitude=.05,alpha=.975,budget=65536,
        source_sha256=digest(__file__),purpose='diagnostic repeated wall timing after main comparison; no accuracy selection',
        non_sampling_runtime='wall estimator minus measurement.sample internal ledger time; includes setup, schedules, posterior processing; not pure posterior CPU time'))
    dist,raw,w=load_inputs();times=[]
    for task in ['amplitude','risk']:
        for seed in range(100,110):
            for method in METHODS:
                tick=perf_counter()
                if task=='amplitude':
                    r=estimate_payload([.95,.05],[0.,1.],method,cfg,65536,seed,epsilon=.005)
                    sampling=r.ledger.runtime_ms
                else:
                    r=estimate_risk(dist,.975,65536,seed,method,replace(cfg,alpha=.975),w)
                    sampling=r['resources']['runtime_ms']
                elapsed=(perf_counter()-tick)*1000
                times.append(dict(task=task,method=method,seed=seed,wall_ms=elapsed,sampling_ms=sampling,
                                  non_sampling_ms=elapsed-sampling))
    tdf=pd.DataFrame(times);tdf.to_csv(DEST/'timing_results.csv',index=False)
    tdf.groupby(['task','method'])[['wall_ms','sampling_ms','non_sampling_ms']].mean().to_csv(DEST/'timing_summary.csv')
    checks['grid_replay_max_error']=float(ps.loc['grid_biqae','max_abs_projection'])
    write_json(DEST/'independent_audit.json',checks)
    write_json(DEST/'checksums.json',{p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':main()
