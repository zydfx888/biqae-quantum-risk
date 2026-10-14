"""Independent reconciliation of full-flow resource artifacts."""
import json
import numpy as np
import pandas as pd
from .config import ROOT, digest, write_json

def main():
    out=ROOT/'artifacts/validation_suite_v4/exp5_resources'
    results=pd.read_csv(out/'results.csv')
    rounds=pd.read_csv(out/'round_resources.csv')
    unique=pd.read_csv(out/'unique_circuits.csv')
    assert len(results)==48 and len(results.drop_duplicates('run_id'))==48
    assert len(results[results.backend=='qiskit_statevector'])==12
    assert not json.loads((out/'failures.json').read_text())
    assert (rounds.weighted_cx==rounds.shots*rounds.circuit_cx).all()
    assert (rounds.weighted_u==rounds.shots*rounds.circuit_u).all()
    assert (rounds.weighted_depth==rounds.shots*rounds.circuit_depth).all()
    assert (rounds.A_inverse_calls==rounds.shots*rounds.k).all()
    assert (rounds.A_forward_calls==rounds.shots*(rounds.k+1)).all()
    assert unique.probability_error.max()<1e-8
    for rid,g in rounds.groupby('run_id'):
        r=results.set_index('run_id').loc[rid]
        assert g.weighted_cx.sum()==r.weighted_cx
        assert g.A_forward_calls.sum()+g.A_inverse_calls.sum()==r.A_calls
        assert r.A_calls<=65536
    pair_count=0
    fields=['stage','query','round','threshold','k','shots','successes','weighted_cx']
    for _,g in rounds[rounds.seed==700].groupby(['states','alpha','method']):
        a=g[g.backend=='analytic_shots'][fields].reset_index(drop=True)
        b=g[g.backend=='qiskit_statevector'][fields].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b); pair_count+=1
    stages=rounds[rounds.backend=='analytic_shots'].groupby(['states','alpha','method','stage']).agg(
        total_cx=('weighted_cx','sum'),A_forward=('A_forward_calls','sum'),A_inverse=('A_inverse_calls','sum'),
        total_shots=('shots','sum'),max_depth=('circuit_depth','max')).reset_index()
    stages['mean_CX_per_run']=stages.total_cx/3
    stages.to_csv(out/'stage_summary.csv',index=False)
    report=out/'实验5报告.md'
    text=report.read_text(encoding='utf-8')
    text+='\n\n## 独立逐轮复核\n\n12 组解析/状态向量配对的全部阈值、轮次、幂、shots、成功计数及 CX 成本逐项完全一致。48 次流程预算均不超过65536，逐轮工作量与总账完全相等。CDF/ES阶段拆分见 stage_summary.csv。\n'
    report.write_text(text,encoding='utf-8')
    checks=dict(runs=len(results),paired_full_round_paths=pair_count,zero_failures=True,
        all_budgets_valid=True,weighted_components_reconciled=True,
        max_probability_error=float(unique.probability_error.max()))
    write_json(out/'independent_verification.json',checks)
    write_json(out/'artifact_hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(checks))

if __name__=='__main__': main()
