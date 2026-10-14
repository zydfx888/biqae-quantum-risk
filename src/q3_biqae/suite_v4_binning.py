"""Exact compression audit before any downstream retraining; frozen weight probes."""
from datetime import datetime, timezone
from time import perf_counter
import argparse
import json
import numpy as np
import pandas as pd
from .config import ROOT, Config, write_json, digest
from .scenario_table import ScenarioTable
from .suite_risk import load_inputs, SOURCE
from .exact_reference import risk, gradient
from .trained_model import freeze_model_scenarios
from .optimizer import project_weights

OUT=ROOT/'artifacts/validation_suite_v4/exp1_binning'
METHODS=['original_equal_tail90','uniform_mass_actual','uniform_width_actual','tail95_actual','tail90_offgrid_actual']

def compress(raw, weights, states, method):
    """Return full-vector conditional means, preserving empirical probability mass."""
    x=np.asarray(raw,dtype=float);w=np.asarray(weights,dtype=float)
    if method=='original_equal_tail90':
        return freeze_model_scenarios(x,states)
    losses=-x@w;order=np.argsort(losses,kind='stable');x=x[order];losses=losses[order]
    n=len(x);p=np.full(n,1/n)
    if method=='uniform_width_actual':
        edges=np.linspace(losses.min(),losses.max(),states+1)
        groups=np.minimum(np.searchsorted(edges,losses,side='right')-1,states-1)
        idx=[np.flatnonzero(groups==i) for i in range(states)]
        idx=[v for v in idx if len(v)]
        return ScenarioTable([x[v].mean(axis=0) for v in idx],[len(v)/n for v in idx])
    if method=='uniform_mass_actual':edges=np.linspace(0,1,states+1)
    elif method=='tail95_actual':
        body=states//2
        edges=np.r_[np.linspace(0,.95,body+1),np.linspace(.95,1,states-body+1)[1:]]
    elif method=='tail90_offgrid_actual':
        body=states//2+1
        edges=np.r_[np.linspace(0,.9,body+1),np.linspace(.9,1,states-body+1)[1:]]
    else:raise ValueError(method)
    # Fractionally split boundary observations; preserve every asset's first moment.
    right=np.arange(1,n+1,dtype=float)/n;left=np.arange(n,dtype=float)/n
    mass=[];centers=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        overlap=np.maximum(0,np.minimum(right,hi)-np.maximum(left,lo))
        m=overlap.sum()
        if m>0:mass.append(m);centers.append(overlap@x/m)
    return ScenarioTable(centers,np.array(mass)/sum(mass))

def exact_metrics(raw_table, reduced, weights, alpha):
    w=np.asarray(weights);x=raw_table.losses(w);z=reduced.losses(w)
    ref=risk(x,raw_table.probabilities,alpha);est=risk(z,reduced.probabilities,alpha)
    g=gradient(raw_table,w,alpha);h=gradient(reduced,w,alpha)
    cg=g-g.mean();ch=h-h.mean()
    denom=max(np.linalg.norm(cg)*np.linalg.norm(ch),1e-18)
    step=.3/max(np.std(x),1e-6)
    update=project_weights(w-step*g,.6)-w;update_h=project_weights(w-step*h,.6)-w
    result={'var_bias_bp':(est['var']-ref['var'])*1e4,'es_bias_bp':(est['es']-ref['es'])*1e4,
        'raw_var':ref['var'],'raw_es':ref['es'],'reduced_var':est['var'],'reduced_es':est['es'],
        'tail_mass_gt_raw_tau_error':float(reduced.probabilities[z>ref['var']].sum()-raw_table.probabilities[x>ref['var']].sum()),
        'tail_mass_ge_raw_tau_error':float(reduced.probabilities[z>=ref['var']].sum()-raw_table.probabilities[x>=ref['var']].sum()),
        'raw_tail_mass_gt':float(raw_table.probabilities[x>ref['var']].sum()),
        'reduced_tail_mass_gt':float(reduced.probabilities[z>ref['var']].sum()),
        'gradient_relative_l2':float(np.linalg.norm(h-g)/max(np.linalg.norm(g),1e-18)),
        'centered_gradient_cosine':float(cg@ch/denom),
        'centered_gradient_relative_l2':float(np.linalg.norm(ch-cg)/max(np.linalg.norm(cg),1e-18)),
        'projected_update_l2_error':float(np.linalg.norm(update_h-update)),
        'gradient_sign_mismatches':int(np.sum(np.sign(g)!=np.sign(h))),
        'centered_gradient_sign_mismatches':int(np.sum(np.sign(cg)!=np.sign(ch))),
        'mean_vector_error_max':float(np.max(np.abs(raw_table.probabilities@raw_table.returns-reduced.probabilities@reduced.returns))),
        'actual_states':len(reduced.probabilities)}
    for j in range(len(g)):
        result.update({f'gradient_raw_{j}':g[j],f'gradient_reduced_{j}':h[j],f'gradient_error_{j}_bp':(h[j]-g[j])*1e4})
    return result

def gate(frame):
    """Engineering tolerances, frozen before observing any compression metrics."""
    return bool((frame.var_bias_bp.abs()<=10).all() and (frame.es_bias_bp.abs()<=5).all()
        and (frame.tail_mass_gt_raw_tau_error.abs()<=.0025).all()
        and (frame.gradient_relative_l2<=.1).all() and (frame.centered_gradient_cosine>=.95).all())

def exact_stage():
    start=perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    _,raw,anchor=load_inputs()
    weights={'lp_anchor':anchor.tolist(),'equal':[.25]*4}
    for j in range(4):
        v=np.full(4,.4/3);v[j]=.6;weights[f'concentrated_{j}']=v.tolist()
        weights[f'rotated_{j}']=np.roll([.4,.3,.2,.1],j).tolist()
    write_json(OUT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'model_sha256':digest(SOURCE/'models/trained_risk_model.npz'),
        'generated_samples_sha256':digest(SOURCE/'generated_scenarios.npz'),
        'methods':METHODS,'states':[16,32,64],'alphas':[.95,.975],'frozen_weights':weights,
        'binning_anchor':'previously trained LP16 weights; all non-anchor probes frozen before computation',
        'uniform_definitions':'both equal probability and equal loss width; empty width bins explicitly dropped and counted',
        'tail95':'half probability bins in [0,.95], half in [.95,1]; alpha boundaries can structurally preserve anchor ES/gradient',
        'offgrid_control':'90% body with states//2+1 bins avoids aligning both alpha levels',
        'gate':{'max_abs_var_bp':10,'max_abs_es_bp':5,'max_abs_tail_probability_error':.0025,'max_relative_gradient_l2':.1,'min_centered_gradient_cosine':.95},
        'training_gate':'Do not run sampled policy retraining before exact gate. If anchor passes but frozen cross-weights fail, test current-weight recompression exactly across every probe before downstream comparison.',
        'retraining_if_gate_passes':{'steps':120,'budget':8192,'learning_rate':.3,'sampling_seeds':[300,301,302],'methods':['mc','grid_biqae'],'states':32,'alpha':.975},
        'warning':'raw truth is finite trained-model sample enumeration, not population or market truth; tail nominal fractional mass always equals 1-alpha and is not used as preservation evidence'})
    full=ScenarioTable(raw,np.full(len(raw),1/len(raw)));rows=[]
    for method in METHODS:
        for states in [16,32,64]:
            reduced=compress(raw,anchor,states,method);reduced.save(OUT/f'{method}_{states}.npz')
            for name,w in weights.items():
                for alpha in [.95,.975]:
                    rows.append({'method':method,'states':states,'weight':name,'alpha':alpha,**exact_metrics(full,reduced,w,alpha)})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'exact_results.csv',index=False)
    summaries=[];gates=[]
    for (method,states),g in frame.groupby(['method','states']):
        for scope,s in [('anchor',g[g.weight=='lp_anchor']),('cross_weight',g[g.weight!='lp_anchor']),('all',g)]:
            summaries.append({'method':method,'states':states,'scope':scope,'n':len(s),
                'var_mae_bp':s.var_bias_bp.abs().mean(),'var_max_abs_bp':s.var_bias_bp.abs().max(),
                'es_mae_bp':s.es_bias_bp.abs().mean(),'es_max_abs_bp':s.es_bias_bp.abs().max(),
                'tail_probability_max_abs_error':s.tail_mass_gt_raw_tau_error.abs().max(),
                'gradient_relative_l2_mean':s.gradient_relative_l2.mean(),'gradient_relative_l2_max':s.gradient_relative_l2.max(),
                'centered_cosine_min':s.centered_gradient_cosine.min(),'projected_update_l2_error_max':s.projected_update_l2_error.max(),
                'passed':gate(s)})
            gates.append({'method':method,'states':states,'scope':scope,'passed':gate(s)})
    pd.DataFrame(summaries).to_csv(OUT/'exact_summary.csv',index=False)
    # Independently check exact current-weight recompression before sampling training.
    dynamic=[]
    for name,w in weights.items():
        for alpha in [.95,.975]:
            reduced=compress(raw,w,32,'tail95_actual')
            dynamic.append({'weight':name,'alpha':alpha,**exact_metrics(full,reduced,w,alpha)})
    pd.DataFrame(dynamic).to_csv(OUT/'current_weight_exact_probe.csv',index=False)
    write_json(OUT/'exact_gate.json',{'frozen_compression':gates,'current_weight_32_passed':gate(pd.DataFrame(dynamic)),
        'elapsed_seconds':perf_counter()-start,'rows':len(frame),'current_weight_probe_rows':len(dynamic),
        'source_sha256':digest(__file__)})
    print(OUT/'exact_gate.json',flush=True)

def adaptive_gate():
    """Registered follow-up after the alpha-aligned 32-state VaR gate failed."""
    path=OUT/'adaptive_followup_protocol.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,{'created_utc':datetime.now(timezone.utc).isoformat(),
        'reason':'Static actual-weight compression failed cross-weight probes; tail95 current-weight32 preserves ES structurally but fails VaR tolerance.',
        'candidate':'tail90_offgrid_actual, 64 states, recompress at each evaluated current weight',
        'gate_unchanged':json.loads((OUT/'protocol.json').read_text())['gate'],
        'weights_unchanged':True,'sampled_training_not_started':True,
        'if_pass':'120-step paired downstream training: raw exact reference, original64/static_actual64/adaptive_actual64; exact, MC and gridBIQAE with seeds300..302 for sampling, lr0.3, budget8192. Recompression time and raw reads reported separately.'})
    _,raw,_=load_inputs();full=ScenarioTable(raw,np.full(len(raw),1/len(raw)))
    protocol=json.loads((OUT/'protocol.json').read_text());rows=[]
    for name,w in protocol['frozen_weights'].items():
        for alpha in [.95,.975]:
            reduced=compress(raw,w,64,'tail90_offgrid_actual')
            rows.append({'weight':name,'alpha':alpha,**exact_metrics(full,reduced,w,alpha)})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'adaptive64_exact_probe.csv',index=False)
    write_json(OUT/'adaptive64_gate.json',{'passed':gate(frame),'rows':len(frame),
        'max_var_abs_bp':float(frame.var_bias_bp.abs().max()),'max_es_abs_bp':float(frame.es_bias_bp.abs().max()),
        'max_tail_probability_abs_error':float(frame.tail_mass_gt_raw_tau_error.abs().max()),
        'max_relative_gradient_l2':float(frame.gradient_relative_l2.max()),'min_centered_cosine':float(frame.centered_gradient_cosine.min()),
        'source_sha256':digest(__file__)})
    print(OUT/'adaptive64_gate.json',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['exact','adaptive_gate']);args=parser.parse_args()
    exact_stage() if args.stage=='exact' else adaptive_gate()
