"""Audit saved policy distances and convergence against independently solved LPs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,hstack,eye
from .config import ROOT,write_json,digest
from .suite_risk import load_inputs
from .suite_v4_binning import compress
from .scenario_table import ScenarioTable
from .exact_reference import cvar

OUT=ROOT/'artifacts/optimization_acceptance'

def solve_sparse(table):
    n,d=table.returns.shape
    obj=np.r_[np.zeros(d),1,table.probabilities/.025]
    aub=hstack([csr_matrix(-table.returns),csr_matrix(-np.ones((n,1))),-eye(n)],format='csr')
    eq=csr_matrix((np.ones(d),(np.zeros(d,dtype=int),np.arange(d))),shape=(1,n+d+1))
    r=linprog(obj,A_ub=aub,b_ub=np.zeros(n),A_eq=eq,b_eq=[1],bounds=[(0,.6)]*d+[(None,None)]+[(0,None)]*n,method='highs')
    assert r.success,r.message
    assert abs(cvar(table,r.x[:d],.975)-r.fun)<1e-8
    return {'weights':r.x[:d].tolist(),'objective':float(r.fun),'success':True,'solver':'scipy_highs_sparse'}

def run():
    OUT.mkdir(exist_ok=False)
    write_json(OUT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'sources':'saved v3 exp1 policies and v4 binning propagation policies; no retraining',
        'acceptance':{'objective_gap_bp':1,'LP_weight_l2':.05,'constraints':1e-8},
        'meaning':'new engineering thresholds fixed before this audit, not previously preregistered training criteria; LP distance alone does not prove uniqueness',
        'reference':'same-model LP; frozen16 for original training and full20003 empirical model for compression comparisons; never optimize heldout test set'})
    frozen,raw,anchor=load_inputs();full=ScenarioTable(raw,np.full(len(raw),1/len(raw)))
    refs={'frozen16':solve_sparse(frozen),'raw20003':solve_sparse(full)}
    write_json(OUT/'lp_references.json',refs)
    datasets=[('v3_frozen16',ROOT/'artifacts/validation_suite_v3/exp1_training/models',frozen,refs['frozen16']),
              ('v4_raw20003',ROOT/'artifacts/validation_suite_v4/exp1_binning/propagation',full,refs['raw20003'])]
    rows=[];traces=[]
    for dataset,folder,table,ref in datasets:
        paths=sorted(folder.glob('policy_*.json'))
        for path in paths:
            policy=json.loads(path.read_text());trajectory=policy.get('trajectory')
            if not trajectory:continue
            w=np.array(policy['weights']);lpw=np.array(ref['weights']);representation=policy.get('representation','original16')
            points=[]
            for step,t in enumerate(trajectory):
                wa=np.array(t.get('averaged_weights',t.get('weights')))
                objective=cvar(table,wa,.975);gap=(objective-ref['objective'])*1e4
                assert gap>=-1e-6
                item={'dataset':dataset,'representation':representation,'method':policy['method'],'seed':policy['seed'],
                    'step':step+1,'objective':objective,'gap_bp':gap,'weights_l2_lp':float(np.linalg.norm(wa-lpw))}
                traces.append(item);points.append(item)
            gap=(cvar(table,w,.975)-ref['objective'])*1e4;dist=float(np.linalg.norm(w-lpw))
            violation=float(abs(w.sum()-1)+np.maximum(-w,0).sum()+np.maximum(w-.6,0).sum())
            final20=np.array([p['gap_bp'] for p in points[-20:]])
            rows.append({'dataset':dataset,'representation':representation,'method':policy['method'],'seed':policy['seed'],
                'source':str(path.relative_to(ROOT)),'source_sha256':digest(path),'objective':cvar(table,w,.975),
                'gap_bp':gap,'weights_l2_lp':dist,'constraint_violation':violation,
                'gap_pass':gap<=1,'weights_pass':dist<=.05,'joint_pass':gap<=1 and dist<=.05 and violation<=1e-8,
                'first_gap_bp':points[0]['gap_bp'],'last20_gap_range_bp':float(np.ptp(final20)),
                'last20_increase_count':int(np.sum(np.diff(final20)>1e-9))})
    data=pd.DataFrame(rows);trace=pd.DataFrame(traces)
    data.to_csv(OUT/'results.csv',index=False);trace.to_csv(OUT/'trajectories.csv',index=False)
    data.groupby(['dataset','representation','method']).agg(n=('gap_bp','size'),mean_gap_bp=('gap_bp','mean'),max_gap_bp=('gap_bp','max'),
        mean_weights_l2_lp=('weights_l2_lp','mean'),max_weights_l2_lp=('weights_l2_lp','max'),joint_pass_rate=('joint_pass','mean'),
        mean_first_gap_bp=('first_gap_bp','mean'),mean_last20_gap_range_bp=('last20_gap_range_bp','mean'),
        max_constraint_violation=('constraint_violation','max')).to_csv(OUT/'summary.csv')
    write_json(OUT/'completion.json',{'policies':len(data),'trajectory_points':len(trace),'status':'complete','source_sha256':digest(__file__)})
    print(OUT,flush=True)

if __name__=='__main__':run()
