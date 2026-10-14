"""Preregistered 100-seed, four-estimator, one-factor risk controls."""
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
import json
import pandas as pd
from .config import ROOT, digest, write_json
from .suite_risk import load_inputs, configure, result_row, cp
from .exact_reference import risk
from .es_estimation import estimate_risk

DEST = ROOT/'artifacts/validation_suite_v5/controls'
METHODS = ['iqae','grid_biqae','normal_biqae','beta_biqae']

def main():
    started=perf_counter(); cfg=configure()
    settings=[('baseline',cfg),('maxk7',replace(cfg,max_grover_power=7)),
              ('maxk127',replace(cfg,max_grover_power=127)),
              ('var20',replace(cfg,var_budget_fraction=.2)),
              ('var60',replace(cfg,var_budget_fraction=.6))]
    DEST.mkdir(parents=True,exist_ok=False)
    write_json(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        methods=METHODS,alphas=[.95,.975],seeds=list(range(100,200)),budget=65536,
        settings={k:c.to_dict() for k,c in settings},design='one factor at a time; all 4000 trials rerun; no selection',
        success='both absolute VaR and ES errors <=5 or10bp relative to frozen16-state exact reference',
        source_hashes={p.name:digest(p) for p in (ROOT/'src/q3_biqae').glob('*.py')},
        model_sha256=digest(ROOT/'artifacts/feasibility_trained_v2/models/trained_risk_model.npz'),
        table_sha256=digest(ROOT/'artifacts/feasibility_trained_v2/frozen_distribution.npz'),
        policy_sha256=digest(ROOT/'artifacts/feasibility_trained_v2/models/policy_lp.json'),
        interval='BIQAE engineering credible intervals; empirical coverage measured',
        multiplicity='CP95 marginal and Bonferroni simultaneous over40 configurations per target',
        backend='analytic finite-shots; expanded A/A-dagger query accounting, not hardware elapsed time'))
    dist,raw,w=load_inputs(); rows=[];logs=[]
    for label,c in settings:
        for alpha in [.95,.975]:
            truth=risk(dist.losses(w),dist.probabilities,alpha)
            for method in METHODS:
                for seed in range(100,200):
                    key=dict(config=label,alpha=alpha,method=method,seed=seed,budget=65536)
                    row=key.copy()
                    try:
                        r=estimate_risk(dist,alpha,65536,seed,method,replace(c,alpha=alpha),w)
                        row.update(result_row(r,truth))
                        rounds=[x for q in r['logs'] for x in q['rounds']]
                        cost=sum(x['shots']*(2*x['k']+1) for x in rounds)
                        assert cost==row['A_calls'] and cost<=65536
                        assert sum(x['shots'] for x in rounds)==row['shots']
                        assert max((x['k'] for x in rounds),default=0)<=c.max_grover_power
                        stage_cost={stage:sum(x['shots']*(2*x['k']+1) for q in r['logs'] if q['stage']==stage for x in q['rounds']) for stage in ['var','es']}
                        row.update(max_actual_k=max((x['k'] for x in rounds),default=0),
                            mean_shot_weighted_k=sum(x['shots']*x['k'] for x in rounds)/max(1,row['shots']),
                            var_A_calls=stage_cost['var'],es_A_calls=stage_cost['es'],
                            round_count=len(rounds),threshold_queries=sum(q['stage']=='var' for q in r['logs']),
                            unused_budget=65536-cost,converged=r['converged'])
                        logs.append({**key,'rounds_json':json.dumps(r['logs'],allow_nan=False)})
                    except Exception as exc:
                        row.update(status='failed',error=repr(exc))
                    rows.append(row)
                pd.DataFrame(rows).to_csv(DEST/'results.csv',index=False)
                print(label,alpha,method,len(rows),round(perf_counter()-started,2),flush=True)
    pd.DataFrame(logs).to_parquet(DEST/'rounds.parquet',index=False,compression='zstd')
    df=pd.DataFrame(rows); summary=[]
    for (label,alpha,method),g in df.groupby(['config','alpha','method']):
        for target in [5,10]:
            good=(g.status=='ok')&(g.var_error_bp<=target)&(g.es_error_bp<=target)
            h=int(good.sum());n=len(g)
            lo,hi=cp(h,n)
            summary.append(dict(config=label,alpha=alpha,method=method,target_bp=target,n=n,
                failures=int((g.status!='ok').sum()),successes=h,success_rate=h/n,
                success_cp_low=lo,success_cp_high=hi,success_simultaneous_cp_low=cp(h,n,.05/40)[0],
                var_mae_bp=g.var_error_bp.mean(),es_mae_bp=g.es_error_bp.mean(),
                var_coverage=g.var_covered.mean(),es_coverage=g.es_covered.mean(),
                var_coverage_cp_low=cp(int(g.var_covered.sum()),n)[0],es_coverage_cp_low=cp(int(g.es_covered.sum()),n)[0],
                var_width_bp=g.var_width_bp.mean(),es_width_bp=g.es_width_bp.mean(),
                var_resolved_rate=g.var_resolved.mean(),converged_rate=g.converged.mean(),
                mean_A_calls=g.A_calls.mean(),mean_var_A_calls=g.var_A_calls.mean(),mean_es_A_calls=g.es_A_calls.mean(),
                mean_shots=g.shots.mean(),mean_unused_budget=g.unused_budget.mean(),
                max_actual_k=g.max_actual_k.max(),mean_max_actual_k=g.max_actual_k.mean(),
                mean_shot_weighted_k=g.mean_shot_weighted_k.mean(),mean_threshold_queries=g.threshold_queries.mean(),
                mean_runtime_ms=g.runtime_ms.mean()))
    s=pd.DataFrame(summary);s.to_csv(DEST/'summary.csv',index=False)
    write_json(DEST/'conclusions.json',dict(
        trials=len(df),exceptions=int((df.status!='ok').sum()),ledger_verified_trials=len(logs),
        outcomes=s[s.target_bp==5].to_dict(orient='records'),
        limits=['Exact reference is frozen input; compression error excluded.',
                'Larger allowed Grover powers can deepen circuits and are not free.',
                'This finite one-factor grid does not establish an optimal configuration.',
                'Normal/Beta are engineering posterior projections, not validated paper reproductions.',
                'Individual CP bounds do not certify all configurations simultaneously.']))
    write_json(DEST/'status.json',dict(status='complete',runtime_s=perf_counter()-started,trials=len(df),exceptions=int((df.status!='ok').sum())))
    write_json(DEST/'checksums.json',{str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':
    main()
