"""Preregistered training and chronological retraining validation suite."""
import argparse
from pathlib import Path
import json
import shutil
import sys
from time import perf_counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import ROOT, Config, digest, write_json
from .feasibility import METHODS, policy_train, empirical_table
from .scenario_table import ScenarioTable
from .trained_model import train_mixture, TrainedRiskModel, freeze_model_scenarios
from .data_prepare import parse_french
from .classical_cvar import solve_cvar
from .exact_reference import risk, cvar

BASE = ROOT/'artifacts/validation_suite_v3'
SOURCE = ROOT/'artifacts/feasibility_trained_v2'
CFG = Config(alpha=.975, budget=8192, optimization_steps=120, epsilon=.005,
             grid_size=2049, max_rounds=16, shots_per_round=128,
             max_grover_power=15, record_posterior=False)
WINDOWS = [dict(train=['2000-01-01','2008-12-31'], validation=['2009-01-01','2010-12-31'], test=['2011-01-01','2015-12-31']),
           dict(train=['2000-01-01','2013-12-31'], validation=['2014-01-01','2015-12-31'], test=['2016-01-01','2020-12-31']),
           dict(train=['2000-01-01','2018-12-31'], validation=['2019-01-01','2020-12-31'], test=['2021-01-01','2025-12-31'])]

def now():
    return datetime.now(timezone.utc).isoformat()

def preregister(directory, experiment):
    directory.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=now(), experiment=experiment, config=CFG.to_dict(),
        methods=METHODS, references=['exact','lp','equal_weight'],
        seeds=list(range(200,230 if experiment==1 else 210)), windows=WINDOWS if experiment==4 else 'frozen v2 split',
        classical_sampling='one shared batch of 8192 full return vectors for all gradient components',
        gradient_threshold='same exact VaR and fractional tail atoms for every method; hybrid ideal simulation',
        model_selection='GMM2 EM seeds 13,29,43: highest validation log density only',
        optimizer_selection='common learning rate [.03,.1,.3] chosen by exact-gradient validation ES only' if experiment==4 else 'frozen 0.3',
        discretization='20003 model draws with seed 20260915, 16 states',
        constraints={'sum_weights':1,'lower_bound':0,'upper_bound':.6},
        feasibility_gate='BIQAE mean test ES <=1.05*exact test ES and all constraints <=1e-8; bootstrap upper bound also reported',
        statistics='paired bootstrap over training seeds, 10000 replicates, fixed seed 301; intervals reflect training randomness only, not market uncertainty',
        query_accounting='classical samples and quantum A/A-dagger calls reported separately; both exclude model fitting and exact threshold processing',
        time_limit_seconds=3000 if experiment==1 else 2700,
        raw_sha256=digest(ROOT/'data/raw/5_Industry_Portfolios_daily_CSV.zip'),
        python=sys.version, source_sha256={f.name:digest(f) for f in (ROOT/'src/q3_biqae').glob('*.py')})
    if not (directory/'protocol.json').exists():
        write_json(directory/'protocol.json',protocol)
    return protocol

def evaluate_saved(directory):
    table = ScenarioTable.load(directory/'frozen_distribution.npz')
    model = TrainedRiskModel.load(directory/'models/trained_risk_model.npz')
    generated = empirical_table(np.load(directory/'generated_scenarios.npz')['returns'])
    splits = {n:empirical_table(pd.read_csv(directory/f'{n}_returns.csv',index_col=0).values) for n in ['train','validation','test']}
    files = sorted((directory/'models').glob('policy_*.json'))
    policies = [json.loads(f.read_text()) for f in files]
    refs = {p['method']:np.array(p['weights']) for p in policies if p['method'] in ['exact','lp']}
    ref_es = {m:cvar(splits['test'],w,.975) for m,w in refs.items()}
    rows=[]
    for f,p in zip(files,policies):
        w=np.array(p['weights']); resources=p.get('training_resources',{})
        row=dict(method=p['method'], seed=p['seed'], weights=json.dumps(w.tolist()),
            training_runtime_s=p.get('training_runtime_s',0),training_samples=resources.get('samples',0),
            training_A_calls=resources.get('oracle_A_calls',0),
            threshold_state_reads=p.get('classical_threshold_state_reads',0),
            constraint_violation=float(abs(w.sum()-1)+np.maximum(-w,0).sum()+np.maximum(w-.6,0).sum()),
            weights_l2_exact=float(np.linalg.norm(w-refs['exact'])),weights_l2_lp=float(np.linalg.norm(w-refs['lp'])),
            frozen_es=cvar(table,w,.975),generated_es=cvar(generated,w,.975),policy_sha256=digest(f))
        for n,t in splits.items():
            r=risk(t.losses(w),t.probabilities,.975)
            row[f'{n}_es']=r['es'];row[f'{n}_var']=r['var']
        row['discretization_es_bias_bp']=(row['frozen_es']-row['generated_es'])*1e4
        row['generated_vs_test_es_gap_bp']=(row['generated_es']-row['test_es'])*1e4
        for m in refs:
            row[f'test_gap_{m}_bp']=(row['test_es']-ref_es[m])*1e4
            row[f'test_gap_{m}_relative_pct']=(row['test_es']/ref_es[m]-1)*100
        if p['method'] in ['mc','sobol','is']:
            assert row['training_samples']==8192*120 and p['gradient_sampling']=='shared_vector_batch'
        assert row['constraint_violation']<1e-8
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(directory/'results.csv',index=False)
    summary=[]
    for method,group in frame.groupby('method'):
        vals=group.test_es.to_numpy();idx=np.random.default_rng(301).integers(0,len(vals),(10000,len(vals)))
        means=vals[idx].mean(axis=1);lo,hi=np.quantile(means,[.025,.975])
        summary.append(dict(method=method,n=len(vals),test_es_mean=vals.mean(),test_es_sd=vals.std(ddof=1) if len(vals)>1 else 0,
            test_es_ci_low=lo,test_es_ci_high=hi,test_gap_exact_relative_pct=(vals.mean()/ref_es['exact']-1)*100,
            test_gap_exact_ci_upper_pct=(hi/ref_es['exact']-1)*100,
            test_gap_lp_relative_pct=(vals.mean()/ref_es['lp']-1)*100,
            weights_l2_exact_mean=group.weights_l2_exact.mean(),weights_l2_lp_mean=group.weights_l2_lp.mean(),
            frozen_es_mean=group.frozen_es.mean(),generated_es_mean=group.generated_es.mean(),
            discretization_es_bias_bp_mean=group.discretization_es_bias_bp.mean(),
            generated_vs_test_es_gap_bp_mean=group.generated_vs_test_es_gap_bp.mean(),
            constraint_max=group.constraint_violation.max(),runtime_s_mean=group.training_runtime_s.mean(),
            samples_mean=group.training_samples.mean(),A_calls_mean=group.training_A_calls.mean()))
    pd.DataFrame(summary).to_csv(directory/'summary.csv',index=False)
    paired=[]
    for baseline in ['mc','sobol','is','iqae']:
        joined=frame[frame.method=='grid_biqae'].merge(frame[frame.method==baseline],on='seed',suffixes=('_biqae','_baseline'))
        d=(joined.test_es_biqae-joined.test_es_baseline).to_numpy()*1e4
        idx=np.random.default_rng(301).integers(0,len(d),(10000,len(d)))
        lo,hi=np.quantile(d[idx].mean(axis=1),[.025,.975])
        paired.append(dict(baseline=baseline,n=len(d),biqae_minus_baseline_mean_bp=d.mean(),ci_low_bp=lo,ci_high_bp=hi))
    write_json(directory/'paired_comparisons.json',paired)
    write_json(directory/'model_evaluation.json',dict(model_sha256=digest(directory/'models/trained_risk_model.npz'),
        log_density={n:float(model.log_density(t.returns).mean()) for n,t in splits.items()},
        reloaded_policy_count=len(rows),test_used_for_selection=False))
    return frame

def checksums(directory):
    write_json(directory/'checksums.json',{str(f.relative_to(directory)):digest(f) for f in directory.rglob('*') if f.is_file() and f.name!='checksums.json'})

def experiment1():
    d=BASE/'exp1_training';preregister(d,1);start=perf_counter()
    expected={k.replace('\\','/'):v for k,v in json.loads((SOURCE/'checksums.json').read_text(encoding='utf-8')).items()};reused={}
    names=['frozen_distribution.npz','generated_scenarios.npz','train_returns.csv','validation_returns.csv','test_returns.csv',
           'model_selection.json','optimizer_selection.json','data_manifest.json','models/trained_risk_model.npz']
    names += [str(f.relative_to(SOURCE)) for f in (SOURCE/'models').glob('policy_*')]
    for name in names:
        assert digest(SOURCE/name)==expected[name.replace('\\','/')],name
        target=d/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,target)
        reused[name]=digest(target)
    write_json(d/'reuse_manifest.json',dict(source=str(SOURCE),sha256=reused,timing='200..209 source runtime reused; 210..229 timed in current run'))
    table=ScenarioTable.load(d/'frozen_distribution.npz')
    for method in METHODS:
        for seed in range(210,230):
            if perf_counter()-start>3000: raise TimeoutError('Experiment 1 time budget')
            print(f'exp1 {method} {seed}',flush=True)
            policy_train(table,CFG,method,seed,.3,d/'models'/f'policy_{method}_{seed}.json')
    frame=evaluate_saved(d)
    write_json(d/'status.json',dict(status='complete',completed_utc=now(),wall_seconds=perf_counter()-start,
        sampled_policies=150,newly_trained_policies=100,reused_sampled_policies=50,rows=len(frame)))
    checksums(d)

def experiment4():
    d=BASE/'exp4_windows';preregister(d,4);start=perf_counter()
    raw=ROOT/'data/raw/5_Industry_Portfolios_daily_CSV.zip'
    all_data=parse_french(raw)[['Cnsmr','Manuf','HiTec','Hlth']];frames=[]
    for index,split in enumerate(WINDOWS,1):
        wd=d/f'window_{index}';models=wd/'models';models.mkdir(parents=True,exist_ok=True)
        write_json(wd/'protocol.json',dict(window=index,split=split,parent_protocol_sha256=digest(d/'protocol.json'),created_utc=now()))
        parts={n:all_data.loc[a:b] for n,(a,b) in split.items()}
        assert parts['train'].index.max()<parts['validation'].index.min()<parts['test'].index.min()
        for name,part in parts.items():part.to_csv(wd/f'{name}_returns.csv',index_label='date')
        write_json(wd/'data_manifest.json',dict(raw_sha256=digest(raw),splits={n:dict(rows=len(p),first=str(p.index.min()),last=str(p.index.max())) for n,p in parts.items()}))
        candidates=[]
        for seed in [13,29,43]:
            model,history=train_mixture(parts['train'].values,seed=seed);model.save(models/f'gmm_{seed}.npz')
            write_json(models/f'gmm_{seed}_training.json',history)
            candidates.append(dict(seed=seed,iterations=len(history),validation_log_density=float(model.log_density(parts['validation'].values).mean())))
        best=max(candidates,key=lambda r:r['validation_log_density']);model=TrainedRiskModel.load(models/f"gmm_{best['seed']}.npz")
        model.save(models/'trained_risk_model.npz');write_json(wd/'model_selection.json',dict(candidates=candidates,selected=best))
        scenarios=model.sample(20003,20260915);np.savez_compressed(wd/'generated_scenarios.npz',returns=scenarios)
        table=freeze_model_scenarios(scenarios,16);table.save(wd/'frozen_distribution.npz')
        validation=empirical_table(parts['validation'].values);rates=[]
        for rate in [.03,.1,.3]:
            p=policy_train(table,CFG,'exact',200,rate)
            write_json(models/f'validation_candidate_lr_{rate}.json',p)
            rates.append(dict(learning_rate=rate,validation_es=cvar(validation,p['weights'],.975)))
        rate=min(rates,key=lambda r:r['validation_es'])['learning_rate']
        write_json(wd/'optimizer_selection.json',dict(validation_only=rates,learning_rate=rate))
        lpstart=perf_counter();lp=solve_cvar(table,.975,.6)
        write_json(models/'policy_lp.json',dict(method='lp',seed=0,training_runtime_s=perf_counter()-lpstart,**lp))
        write_json(models/'policy_equal_weight.json',dict(method='equal_weight',seed=0,weights=[.25]*4))
        for method in ['exact',*METHODS]:
            for seed in ([200] if method=='exact' else range(200,210)):
                if perf_counter()-start>2700:raise TimeoutError('Experiment 4 time budget')
                print(f'exp4 window {index} {method} {seed}',flush=True)
                policy_train(table,CFG,method,seed,rate,models/f'policy_{method}_{seed}.json')
        frame=evaluate_saved(wd);frame.insert(0,'window',index);frames.append(frame)
        write_json(wd/'status.json',dict(status='complete',completed_utc=now(),rows=len(frame),learning_rate=rate))
        checksums(wd)
    pd.concat(frames).to_csv(d/'all_results.csv',index=False)
    summaries=[]
    for index in range(1,4):
        s=pd.read_csv(d/f'window_{index}'/'summary.csv');s.insert(0,'window',index);summaries.append(s)
    pd.concat(summaries).to_csv(d/'all_summary.csv',index=False)
    write_json(d/'status.json',dict(status='complete',completed_utc=now(),wall_seconds=perf_counter()-start,
        retrained_gmm_candidates=9,retrained_sampled_policies=150,windows=3))
    checksums(d)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=['1','4','all'],default='all');args=parser.parse_args()
    if args.experiment in ['1','all']:experiment1()
    if args.experiment in ['4','all']:experiment4()
