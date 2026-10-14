"""Small paired retraining only after exact adaptive compression gate passed."""
from dataclasses import replace
from time import perf_counter
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from .suite_v4_binning import OUT, compress, exact_metrics
from .suite_risk import load_inputs, SOURCE
from .config import Config, write_json, digest
from .scenario_table import ScenarioTable
from .resource_ledger import ResourceLedger
from .exact_reference import gradient, tail_weights, cvar
from .optimizer import project_weights
from .classical_gradient import shared_batch_gradient
from .amplitude import estimate_payload

def run():
    assert json.loads((OUT/'adaptive64_gate.json').read_text())['passed']
    dest=OUT/'propagation';dest.mkdir(exist_ok=False)
    start=perf_counter();_,raw,anchor=load_inputs()
    full=ScenarioTable(raw,np.full(len(raw),1/len(raw)))
    scale=max(float(np.std(full.losses(np.full(4,.25)))),1e-6)
    cfg=Config(alpha=.975,budget=8192,optimization_steps=120,epsilon=.005,grid_size=2049,
               max_rounds=16,shots_per_round=128,max_grover_power=15,record_posterior=False)
    write_json(dest/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'gate_sha256':digest(OUT/'adaptive64_gate.json'),'config':cfg.to_dict(),
        'scale':'common raw-model equal-weight loss standard deviation fixed for all optimizers',
        'scale_value':scale,'learning_rate':.3,'sampling_seeds':[300,301,302],
        'representations':['raw_reference','original64','static_actual64','adaptive_actual64'],
        'controls':'failed static compressions retained as diagnostic controls, not promoted as validated schemes',
        'final_policy':'average of 120 projected updates; all serialized policies reloaded before heldout evaluation',
        'test_selection':False,'adaptive_cost':'every update sorts and recompresses all 20003 raw scenarios; classical cost explicitly counted',
        'source_sha256':digest(__file__)})
    static={'original64':compress(raw,anchor,64,'original_equal_tail90'),
            'static_actual64':compress(raw,anchor,64,'tail90_offgrid_actual')}
    evaluations=[];audit_rows=[];logs=[]
    for representation in ['raw_reference','original64','static_actual64','adaptive_actual64']:
        for method in (['exact'] if representation=='raw_reference' else ['exact','mc','grid_biqae']):
            for seed in ([300] if method=='exact' else [300,301,302]):
                print(representation,method,seed,flush=True)
                t0=perf_counter();w=np.full(4,.25);average=np.zeros(4);rng=np.random.default_rng(seed)
                ledger=ResourceLedger();trace=[];recompression_s=0.;recompression_count=0
                for step in range(120):
                    if representation=='adaptive_actual64':
                        t=perf_counter();tab=compress(raw,w,64,'tail90_offgrid_actual')
                        recompression_s+=perf_counter()-t;recompression_count+=1
                    else:tab=full if representation=='raw_reference' else static[representation]
                    if method=='exact':g=gradient(tab,w,cfg.alpha)
                    elif method=='mc':
                        g,_,ld,rl=shared_batch_gradient(tab,tail_weights(tab,w,cfg.alpha),cfg.alpha,cfg.budget,int(rng.integers(2**31)),method)
                        ledger.merge(ld)
                        logs.append({'representation':representation,'method':method,'seed':seed,'step':step,'logs_json':json.dumps(rl)})
                    else:
                        tail=tail_weights(tab,w,cfg.alpha);g=np.zeros(4);rounds=[]
                        for asset in range(4):
                            for sign in (1,-1):
                                value=np.maximum(sign*tab.returns[:,asset],0);bound=float(value.max())
                                if bound==0:continue
                                a=estimate_payload(tab.probabilities,value*tail/bound,method,cfg,cfg.budget//8,int(rng.integers(2**31)),epsilon=.005)
                                g[asset]-=sign*bound*a.estimate/(1-cfg.alpha);ledger.merge(a.ledger)
                                rounds.append({'asset':asset,'sign':sign,'rounds':a.rounds})
                        logs.append({'representation':representation,'method':method,'seed':seed,'step':step,'logs_json':json.dumps(rounds)})
                    if step%10==0 or step==119:
                        audit_rows.append({'representation':representation,'method':method,'seed':seed,'step':step,
                            'evaluation':'current_weights_before_update',**exact_metrics(full,tab,w,cfg.alpha)})
                    w=project_weights(w-(.3/np.sqrt(step+1))*g/scale,.6)
                    average+=w;wa=average/(step+1)
                    trace.append({'step':step,'current_weights':w.tolist(),'averaged_weights':wa.tolist(),
                        'A_calls':ledger.oracle_A_calls,'samples':ledger.samples})
                final=average/120
                policy={'representation':representation,'method':method,'seed':seed,'weights':final.tolist(),
                    'trajectory':trace,'resources':ledger.to_dict(),'runtime_s':perf_counter()-t0,
                    'recompression_s':recompression_s,'training_recompression_count':recompression_count,
                    'training_recompression_raw_vector_reads':recompression_count*len(raw),
                    'classical_threshold_state_reads':120*(len(raw) if representation=='raw_reference' else 64)}
                path=dest/f'policy_{representation}_{method}_{seed}.json';write_json(path,policy)
                saved=json.loads(path.read_text());final=np.array(saved['weights'])
                # Test is read solely for evaluation, after policy is fixed and saved.
                test_x=pd.read_csv(SOURCE/'test_returns.csv',index_col=0).values
                test=ScenarioTable(test_x,np.full(len(test_x),1/len(test_x)))
                tab=compress(raw,final,64,'tail90_offgrid_actual') if representation=='adaptive_actual64' else full if representation=='raw_reference' else static[representation]
                metric=exact_metrics(full,tab,final,cfg.alpha)
                audit_rows.append({'representation':representation,'method':method,'seed':seed,'step':120,'evaluation':'final_averaged_policy',**metric})
                evaluations.append({'representation':representation,'method':method,'seed':seed,'policy_sha256':digest(path),
                    'raw_model_es':cvar(full,final,cfg.alpha),'test_es':cvar(test,final,cfg.alpha),
                    'runtime_s':saved['runtime_s'],'recompression_s':recompression_s,'recompression_raw_reads':recompression_count*len(raw),
                    'samples':ledger.samples,'A_calls':ledger.oracle_A_calls,
                    'constraint_violation':float(abs(final.sum()-1)+np.maximum(-final,0).sum()+np.maximum(final-.6,0).sum()),**metric})
                pd.DataFrame(evaluations).to_csv(dest/'results.csv',index=False)
    data=pd.DataFrame(evaluations);pd.DataFrame(audit_rows).to_csv(dest/'trajectory_exact_audit.csv',index=False)
    pd.DataFrame(logs).to_parquet(dest/'rounds.parquet',index=False,compression='zstd')
    columns=['raw_model_es','test_es','es_bias_bp','var_bias_bp','gradient_relative_l2','centered_gradient_cosine',
             'runtime_s','recompression_s','recompression_raw_reads','samples','A_calls','constraint_violation']
    data.groupby(['representation','method'])[columns].mean().to_csv(dest/'summary.csv')
    write_json(dest/'status.json',{'status':'complete','policies':len(data),'sampled_policies':int((data.method!='exact').sum()),
        'runtime_s':perf_counter()-start,'test_never_used_for_selection':True,
        'max_constraint_violation':float(data.constraint_violation.max())})
    print(dest,flush=True)

if __name__=='__main__':run()
