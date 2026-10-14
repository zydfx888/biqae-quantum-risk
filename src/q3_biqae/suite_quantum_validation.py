"""Frozen, reproducible implementation cross-validation; no algorithm modifications."""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json
import traceback
import numpy as np
import pandas as pd
from .config import ROOT, Config, digest, write_json
from .amplitude import estimate_payload
from .trained_model import TrainedRiskModel, freeze_model_scenarios
from .exact_reference import risk

OUT = ROOT / 'artifacts/validation_suite_v3/exp5_quantum'
SOURCE = ROOT / 'artifacts/feasibility_trained_v2'


def main():
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    amplitudes = [.01, .025, .05, .3, .5, .9, .99]
    seeds = list(range(500, 530))
    config = Config(epsilon=.005, budget=65536, max_grover_power=31,
                    max_rounds=32, shots_per_round=256, record_posterior=False)
    source_files = [SOURCE/'models/trained_risk_model.npz', SOURCE/'generated_scenarios.npz',
                    SOURCE/'models/policy_lp.json', SOURCE/'protocol.json']
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    sources.update({str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'src/q3_biqae').glob('*.py')})
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=config.to_dict(),
        amplitudes=amplitudes, analytic_seeds=seeds, official_seeds=seeds[:3], circuit_seeds=seeds[:3],
        grids=[2049,8193], priors=[[1,1],[.5,.5],[8,2]], states=[8,16], circuit_powers=[0,1,3,7],
        circuit_alphas=[.95,.975], source_hashes=sources,
        implementation='Existing engineering grid BIQAE and bounded local IQAE, unchanged.',
        prior_rule='All prespecified priors run for every amplitude; no selection using truth.',
        intervals='Local IQAE CP alpha spending; BIQAE posterior credible intervals with delta/32. Empirical coverage is not a frequentist guarantee.',
        official='Qiskit IQAE epsilon=.005 alpha=.05 shots=256; no hard budget or power cap; different stopping and alpha allocation. Cross-validation only, not matched query superiority.',
        queries='Local: shots*(2*k+1) A/A-dagger. Official native num_oracle_queries=sum(shots*k); also report converted total A calls.',
        circuits='Actual Qiskit statevector evolution of probability loading + fused controlled payload rotation + A inverse and Grover. Finite binomial shots from evolved probability, no noise/hardware. Lookup/rotation fused, no explicit loss arithmetic register.',
        gates='Compiled u/cx depth and gates; representative QPY plus all powers actually used.',
        fail_rule='Retain every failed run with exception/trace; no reruns or seed selection.')
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Existing protocol: choose a new output rather than overwrite execution')
    write_json(OUT/'protocol.json', protocol)
    results, logs, failures = [], [], []

    def record(run_id, section, truth, method, seed, result, seconds, **meta):
        row = dict(run_id=run_id, section=section, amplitude=float(truth), method=method, seed=seed,
                   estimate=result.estimate, low=result.interval[0], high=result.interval[1],
                   absolute_error=abs(result.estimate-truth), covered=result.interval[0]<=truth<=result.interval[1],
                   width=result.interval[1]-result.interval[0], success=abs(result.estimate-truth)<=config.epsilon,
                   converged=bool(result.converged), interval_kind=result.interval_kind, seconds=seconds,
                   **result.ledger.to_dict(), **meta)
        results.append(row)
        for i, r in enumerate(result.rounds):
            r = dict(r)
            if r.get('raw_counts') is None and 'successes' in r:
                r['raw_counts'] = {'1':r['successes'], '0':r['shots']-r['successes']}
            logs.append(dict(run_id=run_id, round=i, **r))

    def failure(run_id):
        failures.append(dict(run_id=run_id, traceback=traceback.format_exc()))
        write_json(OUT/'failures.json', failures)

    for a in amplitudes:
        variants = [('iqae',8193,[1,1])] + [('grid_biqae',g,p) for g in [2049,8193] for p in [[1,1],[.5,.5],[8,2]]]
        for method, grid, prior in variants:
            cfg = Config(**{**config.to_dict(), 'grid_size':grid, 'prior':prior})
            for seed in seeds:
                run_id = f'known_{a}_{method}_{grid}_{prior[0]}_{prior[1]}_{seed}'
                tick=perf_counter()
                try:
                    r=estimate_payload([1-a,a],[0,1],method,cfg,cfg.budget,seed)
                    record(run_id,'known',a,method,seed,r,perf_counter()-tick,grid=grid,prior=str(prior),states=2)
                except Exception:
                    failure(run_id)
        print(f'Known amplitude {a} complete: {len(results)} results',flush=True)
    pd.DataFrame(results).to_csv(OUT/'results_partial.csv',index=False)

    from qiskit import QuantumCircuit, qpy, transpile
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

    class RecordingSampler(StatevectorSampler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.raw=[]
        def run(self, pubs, *, shots=None):
            job=super().run(pubs, shots=shots)
            for entry in job.result():
                bits=getattr(entry.data,next(iter(entry.data.keys())))
                self.raw.append(dict(raw_counts=bits.get_counts(), shots=bits.num_shots))
            return job

    for a in amplitudes:
        for seed in seeds[:3]:
            run_id=f'official_{a}_{seed}'
            tick=perf_counter()
            try:
                qc=QuantumCircuit(1)
                qc.ry(2*np.arcsin(np.sqrt(a)),0)
                sampler=RecordingSampler(default_shots=256,seed=seed)
                r=IterativeAmplitudeEstimation(.005,.05,sampler=sampler).estimate(EstimationProblem(qc,objective_qubits=[0]))
                low,high=map(float,r.confidence_interval)
                total=sum(x['shots']*(2*k+1) for x,k in zip(sampler.raw,r.powers[1:]))
                results.append(dict(run_id=run_id,section='official',amplitude=a,method='official_iqae',seed=seed,
                    estimate=r.estimation,low=low,high=high,absolute_error=abs(r.estimation-a),
                    covered=low<=a<=high,width=high-low,success=abs(r.estimation-a)<=.005,
                    converged=(high-low)/2<=.005, interval_kind='qiskit_iqae',seconds=perf_counter()-tick,
                    oracle_A_calls=total,official_native_queries=r.num_oracle_queries,
                    max_grover_power=max(r.powers), shots_total=sum(x['shots'] for x in sampler.raw),grid=0,prior='none',states=2))
                for i,(raw,k) in enumerate(zip(sampler.raw,r.powers[1:])):
                    logs.append(dict(run_id=run_id,round=i,k=k,amplitude_interval=r.estimate_intervals[i+1],**raw))
            except Exception:
                failure(run_id)
        print(f'Official amplitude {a} complete',flush=True)

    from qiskit.quantum_info import Statevector
    from .tail_oracle import TailOracle
    model=TrainedRiskModel.load(SOURCE/'models/trained_risk_model.npz')
    with np.load(SOURCE/'generated_scenarios.npz') as data:
        generated=data['returns']
    regenerated=model.sample(len(generated),20260915)
    if not np.array_equal(generated,regenerated):
        raise RuntimeError('Model scenario regeneration mismatch')
    weights=np.array(json.loads((SOURCE/'models/policy_lp.json').read_text())['weights'])
    circuit_audit=[]
    for states, alpha in [(8,.95),(8,.975),(16,.95),(16,.975)]:
        table=freeze_model_scenarios(regenerated,states)
        table.save(OUT/f'frozen_{states}.npz')
        losses=table.losses(weights)
        tau=risk(losses,table.probabilities,alpha)['var']
        excess=np.maximum(losses-tau,0)
        payload=excess/max(excess) if max(excess)>0 else np.zeros_like(excess)
        a=float(table.probabilities@payload)
        oracle=TailOracle(table.probabilities,payload)
        audit=dict(states=states, alpha=alpha, amplitude=a,table_hash=table.sha256,tau=tau,**oracle.validate(),powers=[])
        for k in [0,1,3,7]:
            qc=oracle.amplified(k)
            sv=Statevector.from_instruction(qc)
            actual=float(sv.probabilities([oracle.n])[1])
            expected=float(np.sin((2*k+1)*np.arcsin(np.sqrt(a)))**2)
            compiled=transpile(qc,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=0)
            with (OUT/f'circuit_{states}_{alpha}_k{k}.qpy').open('wb') as f:
                qpy.dump(qc,f)
            audit['powers'].append(dict(k=k,actual=actual,analytic=expected,error=abs(actual-expected),
                    qubits=qc.num_qubits,depth=compiled.depth(),gates=dict(compiled.count_ops())))
        circuit_audit.append(audit)
        for method in ['iqae','grid_biqae']:
            for seed in seeds[:3]:
                for backend in ['analytic_shots','qiskit_statevector']:
                    run_id=f'circuit_{states}_{alpha}_{method}_{seed}_{backend}'
                    cfg=Config(**{**config.to_dict(),'backend':backend})
                    tick=perf_counter()
                    try:
                        r=estimate_payload(table.probabilities,payload,method,cfg,cfg.budget,seed)
                        record(run_id,'circuit',a,method,seed,r,perf_counter()-tick,states=states,alpha=alpha,grid=8193,prior='[1, 1]',backend=backend)
                        if backend=='qiskit_statevector':
                            for k in sorted({x['k'] for x in r.rounds}):
                                path=OUT/f'circuit_{states}_{alpha}_k{k}.qpy'
                                if not path.exists():
                                    with path.open('wb') as f:
                                        qpy.dump(oracle.amplified(k),f)
                    except Exception:
                        failure(run_id)
        print(f'Circuit states {states} complete',flush=True)
    write_json(OUT/'circuit_audit.json',circuit_audit)
    write_json(OUT/'failures.json',failures)
    write_json(OUT/'round_logs.json',logs)
    df=pd.DataFrame(results)
    df.to_csv(OUT/'results.csv',index=False)
    group=['section','method','grid','prior','amplitude']
    summary=df.groupby(group,dropna=False).agg(runs=('seed','size'),mae=('absolute_error','mean'),
        max_error=('absolute_error','max'),coverage=('covered','mean'),mean_width=('width','mean'),
        success_rate=('success','mean'),convergence_rate=('converged','mean'),mean_A_calls=('oracle_A_calls','mean'),
        mean_seconds=('seconds','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for (states,alpha,method,seed),g in df[df.section=='circuit'].groupby(['states','alpha','method','seed']):
        if len(g)==2:
            paired.append(dict(states=int(states),alpha=alpha,method=method,seed=int(seed),
                 estimate_difference=float(g.estimate.max()-g.estimate.min()),
                 query_difference=float(g.oracle_A_calls.max()-g.oracle_A_calls.min())))
    write_json(OUT/'paired_circuit_check.json',paired)
    aggregate=df.groupby(['section','method','grid','prior'],dropna=False).agg(
        runs=('seed','size'),mae=('absolute_error','mean'),max_error=('absolute_error','max'),
        coverage=('covered','mean'),mean_width=('width','mean'),success_rate=('success','mean'),
        convergence_rate=('converged','mean'),mean_A_calls=('oracle_A_calls','mean')).reset_index()
    aggregate.to_csv(OUT/'aggregate.csv',index=False)
    final=dict(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter()-started,
        results=len(df), failures=len(failures), sections=df.section.value_counts().to_dict(),
        model_regeneration_exact=True, protocol_sha256=digest(OUT/'protocol.json'),
        source_hashes_unchanged=all(digest(ROOT/p)==h for p,h in sources.items()),
        maximum_circuit_probability_error=max(x['error'] for a in circuit_audit for x in a['powers']),
        maximum_inverse_error=max(a['inverse_error'] for a in circuit_audit),
        maximum_paired_estimate_difference=max(p['estimate_difference'] for p in paired))
    write_json(OUT/'completion.json',final)
    print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
