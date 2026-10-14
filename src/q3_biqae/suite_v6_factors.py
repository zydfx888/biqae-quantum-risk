"""Staged factor coverage: dimensions, extreme tails, dependence and stress."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
from time import perf_counter
import json
import numpy as np
import pandas as pd
from .config import ROOT,write_json,digest
from .distributions import generate
from .trained_model import freeze_model_scenarios
from .scenario_table import ScenarioTable
from .exact_reference import risk
from .suite_risk import load_inputs,configure,estimate,result_row,cp
from .amplitude import estimate_payload

OUT=ROOT/'artifacts/validation_suite_v6/factors'
METHODS=['mc','sobol','is','iqae','grid_biqae']

def worker(spec):
    start=perf_counter();dest=OUT/spec['id'];dest.mkdir(exist_ok=False)
    n=20003;d=spec['assets'];base=generate(spec['kind'],n,d,20260916,rho=spec['rho'],df=spec['df'])
    with np.load(ROOT/'artifacts/feasibility_trained_v2/generated_scenarios.npz') as z:old=z['returns']
    # Fixed synthetic extension of learned scales; not six/eight observed industries.
    old_mean=old.mean(axis=0);old_sd=old.std(axis=0)
    mean=np.interp(np.linspace(0,3,d),np.arange(4),old_mean)
    sd=np.interp(np.linspace(0,3,d),np.arange(4),old_sd)
    raw=(base.returns-np.linspace(.0001,.0005,d))/np.linspace(.008,.016,d)*sd+mean
    weights=1/sd;weights/=weights.sum()
    full=ScenarioTable(raw,np.full(n,1/n));table=freeze_model_scenarios(raw,spec['states'])
    np.savez_compressed(dest/'raw.npz',returns=raw,weights=weights,mean_scale=mean,sd_scale=sd)
    table.save(dest/'frozen.npz');truths=[];rows=[];logs=[]
    for alpha in [.95,.975,.99]:
        reference=risk(table.losses(weights),table.probabilities,alpha)
        original=risk(full.losses(weights),full.probabilities,alpha)
        truths.append({**spec,'alpha':alpha,'var':reference['var'],'es':reference['es'],
            'raw_var':original['var'],'raw_es':original['es'],'var_bias_bp':(reference['var']-original['var'])*1e4,
            'es_bias_bp':(reference['es']-original['es'])*1e4})
        seeds=range(100,200) if alpha==.99 else range(100,130)
        for method in METHODS:
            for seed in seeds:
                key={**spec,'alpha':alpha,'method':method,'seed':seed,'budget':65536}
                try:
                    result=estimate(table,weights,alpha,65536,seed,method,replace(configure(),alpha=alpha,assets=d,states=spec['states']))
                    row={**key,**result_row(result,reference),
                        'raw_var_error_bp':abs(result['var']-original['var'])*1e4,
                        'raw_es_error_bp':abs(result['es']-original['es'])*1e4}
                    assert row['queries']<=65536
                    if method in ['iqae','grid_biqae']:
                        count=sum(r['shots']*(2*r['k']+1) for q in result['logs'] for r in q['rounds'])
                        assert count==row['A_calls']
                    logs.append({'alpha':alpha,'method':method,'seed':seed,'logs_json':json.dumps(result['logs'])})
                except Exception as e:row={**key,'status':'failed','error':repr(e)}
                rows.append(row)
    pd.DataFrame(rows).to_csv(dest/'results.csv',index=False)
    pd.DataFrame(truths).to_csv(dest/'truths.csv',index=False)
    pd.DataFrame(logs).to_parquet(dest/'rounds.parquet',index=False,compression='zstd')
    write_json(dest/'completion.json',{'rows':len(rows),'failures':sum(r['status']!='ok' for r in rows),'runtime_s':perf_counter()-start})
    return spec['id']

def amplitude_factors():
    rows=[];logs=[];cfg=configure()
    for a in [.01,.05,.5]:
        for eps in [.05,.02,.01,.005]:
            for delta in [.1,.05,.01]:
                for method in ['iqae','grid_biqae','normal_biqae','beta_biqae']:
                    for seed in range(100,130):
                        key={'amplitude':a,'epsilon':eps,'delta':delta,'method':method,'seed':seed}
                        r=estimate_payload([1-a,a],[0.,1.],method,replace(cfg,epsilon=eps,failure_probability=delta),65536,seed)
                        rows.append({**key,'error':abs(r.estimate-a),'covered':r.interval[0]<=a<=r.interval[1],
                            'width':r.interval[1]-r.interval[0],'success':abs(r.estimate-a)<=eps,'A_calls':r.ledger.oracle_A_calls})
                        logs.append({**key,'logs_json':json.dumps(r.rounds)})
    f=pd.DataFrame(rows);f.to_csv(OUT/'amplitude_results.csv',index=False)
    pd.DataFrame(logs).to_parquet(OUT/'amplitude_rounds.parquet',index=False,compression='zstd')
    f.groupby(['epsilon','delta','method']).agg(n=('error','size'),mae=('error','mean'),coverage=('covered','mean'),success=('success','mean'),width=('width','mean'),queries=('A_calls','mean')).to_csv(OUT/'amplitude_summary.csv')

def run():
    start=perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    cases=[]
    for d in [4,6,8]:
        specs=[('normal',rho,5) for rho in [.2,.5,.8]]+[('student_t',.5,df) for df in [3,5,10]]+[('t_copula',.5,5),('regime',.5,5),('extreme',.5,5)]
        for kind,rho,df in specs:cases.append({'id':f'{kind}_d{d}_r{rho}_nu{df}_s64','kind':kind,'assets':d,'rho':rho,'df':df,'states':64})
    for states in [8,16,32,128]:cases.append({'id':f'normal_d4_r0.5_nu5_s{states}','kind':'normal','assets':4,'rho':.5,'df':5,'states':states})
    load_inputs() # verifies trained checkpoint and exact generated samples before synthetic extensions
    write_json(OUT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'cases':cases,
        'alphas':[.95,.975,.99],'seeds_99':list(range(100,200)),'seeds_other':list(range(100,130)),
        'methods':METHODS,'budget':65536,'amplitude_targets':[.05,.02,.01,.005],'failure_probabilities':[.1,.05,.01],
        'amplitude_seeds':list(range(100,130)),'amplitudes':[.01,.05,.5],
        'factor_design':'all requested marginal levels; dimensional extension x distribution family, selected state-size probe; not full Cartesian product',
        'distribution_definitions':'existing multivariate normal/t, t copula with asymmetric monotone margins, 90/10 regimes,2% common minus6sigma shock',
        'training_relation':'synthetic factor returns rescaled using trained-model generated4-asset mean/sd interpolated to6/8; not new observed-asset fits',
        'weights':'inverse learned marginal sd, frozen before sampling','truth':'finite empirical20003 and compressed table enumerated separately',
        'model_sha256':digest(ROOT/'artifacts/feasibility_trained_v2/models/trained_risk_model.npz'),'source_sha256':digest(__file__)})
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,s) for s in cases]
        for future in as_completed(futures):print('completed',future.result(),flush=True)
    frames=[pd.read_csv(OUT/s['id']/'results.csv') for s in cases]
    f=pd.concat(frames,ignore_index=True);f.to_csv(OUT/'results.csv',index=False)
    pd.concat([pd.read_csv(OUT/s['id']/'truths.csv') for s in cases]).to_csv(OUT/'truths.csv',index=False)
    summaries=[]
    for keys,g in f.groupby(['id','kind','assets','rho','df','states','alpha','method']):
        row=dict(zip(['id','kind','assets','rho','df','states','alpha','method'],keys))
        n=len(g);good=g.status=='ok'
        for target in [5,10]:
            h=int((good&(g.var_error_bp<=target)&(g.es_error_bp<=target)).sum())
            summaries.append({**row,'target_bp':target,'n':n,'failures':int((~good).sum()),'successes':h,'success_rate':h/n,
                'cp_low':cp(h,n)[0],'es_mae_bp':g.es_error_bp.mean(),'var_mae_bp':g.var_error_bp.mean(),
                'raw_es_mae_bp':g.raw_es_error_bp.mean(),'raw_success_rate':float((good&(g.raw_var_error_bp<=target)&(g.raw_es_error_bp<=target)).mean()),
                'es_coverage':g.es_covered.fillna(False).mean(),'var_coverage':g.var_covered.fillna(False).mean(),
                'es_width_bp':g.es_width_bp.mean(),'mean_queries':g.queries.mean()})
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    amplitude_factors()
    write_json(OUT/'completion.json',{'status':'complete','risk_trials':len(f),'amplitude_trials':4320,
        'failed_risk_trials':int((f.status!='ok').sum()),'cases':len(cases),'runtime_s':perf_counter()-start})
    write_json(OUT/'checksums.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':run()
