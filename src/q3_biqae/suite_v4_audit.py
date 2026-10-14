"""Reconstruct training query counts and verify all completed supplement artifacts."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import ROOT, digest, write_json

BASE=ROOT/'artifacts/validation_suite_v4'

def run():
    d=BASE/'exp1_binning/propagation';policies=pd.read_csv(d/'results.csv')
    rounds=pd.read_parquet(d/'rounds.parquet');rows=[];max_violation=0.
    assert len(policies)==22 and len(rounds)==2160
    for p in policies.itertuples():
        path=d/f'policy_{p.representation}_{p.method}_{p.seed}.json'
        assert digest(path)==p.policy_sha256
        saved=json.loads(path.read_text());assert len(saved['trajectory'])==120
        arrays=np.array([t['current_weights'] for t in saved['trajectory']])
        average=np.cumsum(arrays,axis=0)/np.arange(1,121)[:,None]
        np.testing.assert_allclose(average,np.array([t['averaged_weights'] for t in saved['trajectory']]),atol=1e-14)
        np.testing.assert_allclose(average[-1],saved['weights'],atol=1e-14)
        violation=np.max(np.abs(arrays.sum(axis=1)-1)+np.maximum(-arrays,0).sum(axis=1)+np.maximum(arrays-.6,0).sum(axis=1))
        max_violation=max(max_violation,float(violation));assert violation<1e-8
        group=rounds[(rounds.representation==p.representation)&(rounds.method==p.method)&(rounds.seed==p.seed)].sort_values('step')
        samples=0;calls=0;shots=0
        if p.method!='exact':assert list(group.step)==list(range(120))
        for r in group.itertuples():
            item=json.loads(r.logs_json);stepcost=0
            for query in item:
                for q in query['rounds']:
                    if p.method=='mc':
                        assert q['assets']==4 and sum(q['state_counts'])==q['samples']==8192
                        samples+=q['samples'];stepcost+=q['samples']
                    else:
                        assert 0<=q['successes']<=q['shots'] and 0<=q['k']<=15
                        calls+=q['shots']*(2*q['k']+1);shots+=q['shots'];stepcost+=q['shots']*(2*q['k']+1)
            assert stepcost<=8192
        assert samples==p.samples and calls==p.A_calls
        assert samples==saved['resources']['samples'] and calls==saved['resources']['oracle_A_calls']
        rows.append({'representation':p.representation,'method':p.method,'seed':p.seed,'samples_reconstructed':samples,'A_calls_reconstructed':calls,'shots_reconstructed':shots})
    pd.DataFrame(rows).to_csv(BASE/'training_ledger_reconstruction.csv',index=False)
    manifest_entries=0
    for path in [*BASE.rglob('checksums.json'),*BASE.rglob('artifact_hashes.json')]:
        for name,value in json.loads(path.read_text(encoding='utf-8')).items():
            assert digest(path.parent/name)==value,(path,name)
            manifest_entries+=1
    counts={'distribution_trials':len(pd.read_csv(BASE/'exp2_distributions/results.csv')),
        'known_amplitude_trials':len(pd.read_csv(BASE/'exp3_variants/amplitude_results.csv')),
        'variant_risk_trials':len(pd.read_csv(BASE/'exp3_variants/risk_results.csv')),
        'control_risk_trials':len(pd.read_csv(BASE/'exp4_controls/risk_results.csv'))}
    assert list(counts.values())==[6000,840,800,840]
    complete=json.loads((BASE/'exp5_resources/completion.json').read_text())
    assert complete['runs']==48 and complete['rounds']==1513 and complete['failures']==0
    write_json(BASE/'final_audit.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),
        'status':'pass','training_policies_reloaded':len(policies),'training_steps_reconstructed':len(rounds),
        'training_ledger_exact_match':True,'max_all_update_constraint_violation':max_violation,
        'manifest_entries_verified':manifest_entries,'completed_experiment_counts':{**counts,'full_resource_flows':48,'full_resource_rounds':1513},
        'source_hashes':{str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'src/q3_biqae').glob('*.py')},
        'files':{str(f.relative_to(BASE)):digest(f) for f in BASE.rglob('*') if f.is_file() and f.name not in ['final_audit.json']}})
    print('Passed:',len(policies),'policies;',len(rounds),'training steps;',manifest_entries,'manifest entries')

if __name__=='__main__':run()
