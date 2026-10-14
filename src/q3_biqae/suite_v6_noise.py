"""Gate-level density-matrix noise, cached per actual amplified query."""
from datetime import datetime, timezone
from time import perf_counter
import json
import numpy as np
import pandas as pd
from qiskit import transpile, qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .config import ROOT, Config, write_json, digest
from .measurement import Measurement
from .tail_oracle import TailOracle
from .trained_model import freeze_model_scenarios
from .exact_reference import risk
from .es_estimation import estimate_risk

OUT=ROOT/'artifacts/validation_suite_v6/noise'
SOURCE=ROOT/'artifacts/feasibility_trained_v2'
CACHE={}; CIRCUITS={}; NOISE=0.; CROWS=[]
class CachedNoiseMeasurement(Measurement):
    def sample(self,k,shots):
        key=(self.p.tobytes(),self.y.tobytes(),k)
        if key not in CIRCUITS:
            tick=perf_counter(); oracle=TailOracle(self.p,self.y)
            qc=transpile(oracle.amplified(k),basis_gates=['u','cx'],optimization_level=1,seed_transpiler=0)
            stats=dict(id=len(CIRCUITS),states=len(self.p),k=k,qubits=qc.num_qubits,depth=qc.depth(),cx=int(qc.count_ops().get('cx',0)),u=int(qc.count_ops().get('u',0)),compile_seconds=perf_counter()-tick)
            CIRCUITS[key]=(qc,oracle.n,stats)
            if k==0:
                with (OUT/f"circuit_{stats['id']}.qpy").open('wb') as f:qpy.dump(qc,f)
        qc,flag,stats=CIRCUITS[key]; nk=(key,NOISE)
        if nk not in CACHE:
            model=NoiseModel()
            if NOISE:
                model.add_all_qubit_quantum_error(depolarizing_error(NOISE,1),['u'])
                model.add_all_qubit_quantum_error(depolarizing_error(NOISE,2),['cx'])
            sim=AerSimulator(method='density_matrix',noise_model=model,max_parallel_threads=1)
            c=qc.copy(); c.save_probabilities([flag],label='flag')
            tick=perf_counter(); prob=float(sim.run(c,shots=1).result().data(0)['flag'][1])
            CACHE[nk]=prob
            theta=np.arcsin(np.sqrt(np.clip(self.p@self.y,0,1)))
            CROWS.append(dict(**stats,noise=NOISE,probability=prob,ideal_probability=float(np.sin((2*k+1)*theta)**2),density_seconds=perf_counter()-tick))
        h=int(self.rng.binomial(shots,np.clip(CACHE[nk],0,1)))
        self.ledger.record(shots,k)
        self.ledger.max_depth=max(self.ledger.max_depth,stats['depth'])
        self.ledger.qubits=stats['qubits']
        self.ledger.max_two_qubit_gates=max(self.ledger.max_two_qubit_gates,stats['cx'])
        self.ledger.total_two_qubit_gates+=shots*stats['cx']
        self.rounds.append(dict(k=k,shots=shots,successes=h,oracle_A_calls_cumulative=self.ledger.oracle_A_calls,noise=NOISE,circuit_id=stats['id'],cx=stats['cx'],u=stats['u'],depth=stats['depth']))
        return h
    def finalize_resources(self):pass

def main():
    global NOISE
    import q3_biqae.measurement as measurement
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Output exists')
    start=perf_counter()
    cfg=Config(epsilon=.0005,budget=16384,max_grover_power=7,max_rounds=32,shots_per_round=256,grid_size=4097,record_posterior=False)
    write_json(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),states=[8,16],alphas=[.95,.975],seeds=list(range(1100,1130)),methods=['iqae','grid_biqae'],noise=[0,.0001,.001,.01],config=cfg.to_dict(),noise_model='Aer density matrix: depolarizing lambda applied after each compiled u and cx gate, including preparation, Grover A and inverse A. No readout noise. Exact noisy marginal cached, binomial independent finite shots per actual adaptive round. Zero noise is ideal circuit finite shots; exact risk reference has no sampling.',source_hash=digest(SOURCE/'models/trained_risk_model.npz')))
    raw=np.load(SOURCE/'generated_scenarios.npz')['returns']; w=np.array(json.loads((SOURCE/'models/policy_lp.json').read_text())['weights'])
    results=[]; logs=[]; refs=[]
    original=measurement.Measurement; measurement.Measurement=CachedNoiseMeasurement
    try:
      for states in [8,16]:
       table=freeze_model_scenarios(raw,states)
       for alpha in [.95,.975]:
        truth=risk(table.losses(w),table.probabilities,alpha); refs.append(dict(states=states,alpha=alpha,**truth))
        for NOISE in [0.,.0001,.001,.01]:
         for method in ['iqae','grid_biqae']:
          for seed in range(1100,1130):
           r=estimate_risk(table,alpha,cfg.budget,seed,method,cfg,w)
           row=dict(states=states,alpha=alpha,noise=NOISE,method=method,seed=seed,var_error_bp=abs(r['var']-truth['var'])*10000,es_error_bp=abs(r['es']-truth['es'])*10000,var_coverage=r['ci']['var'][0]-1e-12<=truth['var']<=r['ci']['var'][1]+1e-12,es_coverage=r['ci']['es'][0]-1e-12<=truth['es']<=r['ci']['es'][1]+1e-12,var_width_bp=np.ptp(r['ci']['var'])*10000,es_width_bp=np.ptp(r['ci']['es'])*10000,A_calls=r['resources']['oracle_A_calls'],shots=r['resources']['shots_total'],cx=r['resources']['total_two_qubit_gates'],max_depth=r['resources']['max_depth'])
           results.append(row); logs.append(dict(**{x:row[x] for x in ['states','alpha','noise','method','seed']},logs=r['logs']))
          print(f'Finished {states} {alpha} {NOISE} {method}',flush=True)
          pd.DataFrame(results).to_csv(OUT/'results.csv',index=False)
    finally:measurement.Measurement=original
    df=pd.DataFrame(results)
    summary=df.groupby(['states','alpha','noise','method']).agg(runs=('seed','size'),var_mae_bp=('var_error_bp','mean'),es_mae_bp=('es_error_bp','mean'),var_coverage=('var_coverage','mean'),es_coverage=('es_coverage','mean'),var_width_bp=('var_width_bp','mean'),es_width_bp=('es_width_bp','mean'),A_calls=('A_calls','mean'),shots=('shots','mean'),cx=('cx','mean'),max_depth=('max_depth','max')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False); pd.DataFrame(CROWS).to_csv(OUT/'circuits.csv',index=False)
    write_json(OUT/'logs.json',logs);write_json(OUT/'exact_reference.json',refs)
    zero=[abs(x['probability']-x['ideal_probability']) for x in CROWS if x['noise']==0]
    write_json(OUT/'completion.json',dict(runs=len(df),elapsed_seconds=perf_counter()-start,unique_circuits=len(CIRCUITS),density_evolutions=len(CROWS),max_zero_noise_probability_error=max(zero),failures=0))
    write_json(OUT/'hashes.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='hashes.json'})
if __name__=='__main__':main()
