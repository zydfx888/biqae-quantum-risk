"""Frozen multi-window validation of dynamic compression, with saved policies."""
import json
from time import perf_counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from .config import ROOT, Config, write_json, digest
from .suite_v4_binning import compress, exact_metrics
from .feasibility import empirical_table
from .trained_model import TrainedRiskModel
from .exact_reference import gradient, tail_weights, cvar
from .optimizer import project_weights
from .classical_gradient import shared_batch_gradient
from .amplitude import estimate_payload
from .resource_ledger import ResourceLedger

OUT=ROOT/'artifacts/validation_suite_v6/stability'
CFG=Config(alpha=.975,budget=8192,optimization_steps=120,epsilon=.005,grid_size=2049,max_rounds=16,shots_per_round=128,max_grover_power=15,record_posterior=False)

def sparse_lp(full):
    n,d=full.returns.shape
    a=sparse.hstack([-sparse.csr_matrix(full.returns),-sparse.csr_matrix(np.ones((n,1))),-sparse.eye(n)],format='csr')
    eq=sparse.csr_matrix(([1.]*d,([0]*d,list(range(d)))),shape=(1,d+1+n))
    res=linprog(np.r_[np.zeros(d),1,full.probabilities/(1-CFG.alpha)],A_ub=a,b_ub=np.zeros(n),A_eq=eq,b_eq=[1],bounds=[(0,.6)]*d+[(None,None)]+[(0,None)]*n,method='highs')
    assert res.success,res.message
    return {'weights':res.x[:d].tolist(),'cvar':float(res.fun),'solver':'scipy_highs_sparse_lp'}

def train(raw, representation, method, seed, rate, path):
    start=perf_counter();full=empirical_table(raw);d=raw.shape[1]
    w=np.full(d,1/d);scale=max(np.std(full.losses(w)),1e-6)
    static=compress(raw,w,64,'original_equal_tail90') if representation=='original64' else full
    average=np.zeros(d);rng=np.random.default_rng(seed);ledger=ResourceLedger();trace=[];logs=[];recompress_s=0.
    for step in range(CFG.optimization_steps):
        if representation=='dynamic64':
            t=perf_counter();tab=compress(raw,w,64,'tail90_offgrid_actual');recompress_s+=perf_counter()-t
        else:tab=static
        if method=='exact':g=gradient(tab,w,CFG.alpha)
        elif method=='mc':
            g,_,ld,rl=shared_batch_gradient(tab,tail_weights(tab,w,CFG.alpha),CFG.alpha,CFG.budget,int(rng.integers(2**31)),method)
            ledger.merge(ld);logs.append({'step':step,'details':json.dumps(rl)})
        else:
            tail=tail_weights(tab,w,CFG.alpha);g=np.zeros(d)
            for asset in range(d):
                for sign in (1,-1):
                    value=np.maximum(sign*tab.returns[:,asset],0);bound=float(value.max())
                    if bound==0:continue
                    a=estimate_payload(tab.probabilities,value*tail/bound,method,CFG,CFG.budget//(2*d),int(rng.integers(2**31)),epsilon=.005)
                    g[asset]-=sign*bound*a.estimate/(1-CFG.alpha);ledger.merge(a.ledger)
                    logs.append({'step':step,'asset':asset,'sign':sign,'details':json.dumps(a.rounds)})
        w=project_weights(w-rate/np.sqrt(step+1)*g/scale,.6);average+=w;wa=average/(step+1)
        trace.append({'step':step+1,'weights':wa.tolist(),'current_weights':w.tolist(),'raw_es':cvar(full,wa,CFG.alpha),'A_calls':ledger.oracle_A_calls,'samples':ledger.samples})
    p={'representation':representation,'method':method,'seed':seed,'learning_rate':rate,'weights':(average/CFG.optimization_steps).tolist(),'trajectory':trace,'resources':ledger.to_dict(),'runtime_s':perf_counter()-start,'recompression_s':recompress_s,'recompression_raw_reads':len(raw)*120 if representation=='dynamic64' else 0}
    write_json(path,p)
    pd.DataFrame(logs).to_parquet(path.with_suffix('.rounds.parquet'),index=False)
    return json.loads(path.read_text())

def run():
    OUT.mkdir(parents=True,exist_ok=False);start=perf_counter()
    write_json(OUT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'config':CFG.to_dict(),'windows':'same three frozen chronological v3 windows','seeds':list(range(600,610)),'representations':['original64','dynamic64'],'methods':['mc','grid_biqae'],'reference':['raw exact gradient','raw LP'],'learning_rate':'reuse common rate previously chosen by validation only in each window','raw_samples':20003,'gate':'report exact probes and final metrics, no test-based exclusion','source_sha256':digest(__file__)})
    rows=[];probes=[]
    for window in [1,2,3]:
        source=ROOT/f'artifacts/validation_suite_v3/exp4_windows/window_{window}';dest=OUT/f'window_{window}';dest.mkdir()
        raw=np.load(source/'generated_scenarios.npz')['returns'];model=TrainedRiskModel.load(source/'models/trained_risk_model.npz')
        assert np.array_equal(raw,model.sample(20003,20260915))
        write_json(dest/'source_manifest.json',{str(p.relative_to(source)):digest(p) for p in [source/'generated_scenarios.npz',source/'models/trained_risk_model.npz',source/'optimizer_selection.json',source/'test_returns.csv']})
        full=empirical_table(raw);lp=sparse_lp(full);write_json(dest/'raw_lp.json',lp)
        test=empirical_table(pd.read_csv(source/'test_returns.csv',index_col=0).values)
        rate=json.loads((source/'optimizer_selection.json').read_text())['learning_rate']
        weights=[np.full(4,.25),np.array(lp['weights'])]+[np.roll([.6,.2,.1,.1],i) for i in range(4)]
        for wi,w in enumerate(weights):
            for alpha in [.95,.975]:probes.append({'window':window,'probe':wi,'alpha':alpha,**exact_metrics(full,compress(raw,w,64,'tail90_offgrid_actual'),w,alpha)})
        pd.DataFrame(probes).to_csv(OUT/'exact_probes.csv',index=False)
        for rep,method,seeds in [('raw','exact',[600])]+[(r,m,range(600,610)) for r in ['original64','dynamic64'] for m in ['mc','grid_biqae']]:
            for seed in seeds:
                print(window,rep,method,seed,flush=True)
                path=dest/f'policy_{rep}_{method}_{seed}.json';p=train(raw,rep,method,seed,rate,path);w=np.array(p['weights'])
                tab=compress(raw,w,64,'tail90_offgrid_actual') if rep=='dynamic64' else compress(raw,w,64,'original_equal_tail90') if rep=='original64' else full
                row={'window':window,'representation':rep,'method':method,'seed':seed,'weights':json.dumps(w.tolist()),'raw_es':cvar(full,w,.975),'test_es':cvar(test,w,.975),'raw_lp_es':lp['cvar'],'test_lp_es':cvar(test,lp['weights'],.975),'lp_gap_bp':(cvar(full,w,.975)-lp['cvar'])*1e4,'lp_weights_l2':float(np.linalg.norm(w-lp['weights'])),'constraint_violation':float(abs(w.sum()-1)+np.maximum(-w,0).sum()+np.maximum(w-.6,0).sum()),'runtime_s':p['runtime_s'],'recompression_s':p['recompression_s'],'recompression_raw_reads':p['recompression_raw_reads'],'samples':p['resources']['samples'],'A_calls':p['resources']['oracle_A_calls'],'policy_sha256':digest(path),**exact_metrics(full,tab,w,.975)}
                rows.append(row);pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
    data=pd.DataFrame(rows);cols=['raw_es','test_es','lp_gap_bp','lp_weights_l2','es_bias_bp','var_bias_bp','gradient_relative_l2','runtime_s','recompression_s','samples','A_calls']
    data.groupby(['window','representation','method'])[cols].agg(['mean','std','max']).to_csv(OUT/'summary.csv')
    paired=[]
    for window in [1,2,3]:
        for method in ['mc','grid_biqae']:
            g=data[(data.window==window)&(data.method==method)].pivot(index='seed',columns='representation',values='test_es');diff=(g.dynamic64-g.original64).values*1e4
            b=np.random.default_rng(701).choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
            paired.append({'window':window,'method':method,'dynamic_minus_original_test_es_bp':float(diff.mean()),'bootstrap95_low_bp':float(np.quantile(b,.025)),'bootstrap95_high_bp':float(np.quantile(b,.975))})
    write_json(OUT/'paired_results.json',paired)
    write_json(OUT/'completion.json',{'status':'complete','trained_policies':len(rows),'sampled_policies':120,'exact_probes':len(probes),'runtime_s':perf_counter()-start,'max_constraint_violation':float(data.constraint_violation.max())})
    write_json(OUT/'checksums.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':run()
