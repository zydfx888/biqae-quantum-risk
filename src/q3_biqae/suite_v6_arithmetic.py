"""Complete risk query ledgers with reversible asset arithmetic."""
import json
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from qiskit import transpile,qpy
from qiskit.quantum_info import Statevector
from .config import ROOT,Config,write_json,digest
from .arithmetic_oracle import ArithmeticOracle
from .trained_model import freeze_model_scenarios
from .distributions import ScenarioTable
from .es_estimation import estimate_risk
from .exact_reference import risk
OUT=ROOT/'artifacts/validation_suite_v6/arithmetic'
SOURCE=ROOT/'artifacts/feasibility_trained_v2'

def stats(qc):
    tick=perf_counter();c=transpile(qc,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=0)
    return c,dict(qubits=c.num_qubits,depth=c.depth(),cx=int(c.count_ops().get('cx',0)),u=int(c.count_ops().get('u',0)),compile_seconds=perf_counter()-tick)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Existing output')
    start=perf_counter();raw=np.load(SOURCE/'generated_scenarios.npz')['returns']
    originals={s:freeze_model_scenarios(raw,s) for s in [8,16,32]}
    amin=min(float((-t.returns).min()) for t in originals.values());amax=max(float((-t.returns).max()) for t in originals.values());unit=(amax-amin)/3
    iw=np.array([2,1,1,2]);w=iw/iw.sum()
    cfg=Config(epsilon=.0005,budget=4096,max_grover_power=3,max_rounds=16,shots_per_round=128,grid_size=2049,record_posterior=False)
    write_json(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),states=[8,16,32],alphas=[.95,.975],seeds=[1300,1301],methods=['iqae','grid_biqae'],config=cfg.to_dict(),weights=w.tolist(),asset_return_bits=2,loss_bits=5,asset_loss_offset=amin,asset_loss_unit=unit,execution='Actual full adaptive ideal finite-shot risk schedules. All actual queries and powers reconstructed with asset qROM, controlled weighted integer addition, explicit loss register, comparator and excess subtraction, encoding and uncomputation; all compiled. Statevector validation on each unique 8-state k=0 circuit plus k=1 CDF and ES. Higher powers verified from unitary amplitude-amplification identity after base circuit validation, not separately statevector simulated.',qualification='2-bit asset fixed point deliberately small demonstration, not production 5bp accuracy; quantization error reported independently. No hardware routing, error correction or physical timing.',source_model_hash=digest(SOURCE/'models/trained_risk_model.npz')))
    rows=[];circuits=[];components=[];validations=[];results=[];logs=[];quant=[];cache={};basecache={}
    for states,original in originals.items():
      codes=np.rint((-original.returns-amin)/unit).astype(int);assert codes.min()>=0 and codes.max()<=3
      table=ScenarioTable(-(amin+unit*codes),original.probabilities)
      codesloss=codes@iw
      for alpha in [.95,.975]:
        truth=risk(table.losses(w),table.probabilities,alpha);old=risk(original.losses(w),original.probabilities,alpha)
        quant.append(dict(states=states,alpha=alpha,var_bias_bp=(truth['var']-old['var'])*10000,es_bias_bp=(truth['es']-old['es'])*10000,max_loss_error_bp=float(np.max(np.abs(table.losses(w)-original.losses(w))))*10000))
        for method in ['iqae','grid_biqae']:
          for seed in [1300,1301]:
            run=f'{states}_{alpha}_{method}_{seed}';r=estimate_risk(table,alpha,cfg.budget,seed,method,cfg,w);logs.append(dict(run=run,logs=r['logs']));runrows=[]
            for query,stage in enumerate(r['logs']):
              threshold=int(round((stage['threshold']-amin)*iw.sum()/unit));bkey=(states,stage['stage'],threshold)
              if bkey not in basecache:
                oracle=ArithmeticOracle(table.probabilities,codes,iw,threshold,stage['stage']);basecache[bkey]=oracle
                for block,qc in oracle.components.items():
                  _,s=stats(qc);components.append(dict(states=states,stage=stage['stage'],threshold=threshold,block=block,**s))
                c,s=stats(oracle.circuit)
                with (OUT/f'A_{states}_{stage["stage"]}_{threshold}.qpy').open('wb') as f:qpy.dump(c,f)
                if states==8:
                  tick=perf_counter();v=Statevector.from_instruction(c);actual=float(v.probabilities([oracle.flag])[1]);expected=float(table.probabilities@oracle.payload)
                  workspace=list(range(oracle.n,oracle.flag));zero=float(v.probabilities(workspace)[0]);error=abs(actual-expected)
                  assert error<1e-8 and abs(zero-1)<1e-8,(bkey,error,zero)
                  restored=v.evolve(c.inverse());inverse_error=float(abs(restored.data[0])**2-1)
                  validations.append(dict(states=states,stage=stage['stage'],threshold=threshold,k=0,probability_error=error,workspace_zero_probability=zero,inverse_probability_error=abs(inverse_error),seconds=perf_counter()-tick))
              oracle=basecache[bkey]
              for rid,rr in enumerate(stage['rounds']):
                k=int(rr['k']);key=bkey+(k,)
                if key not in cache:
                  c,s=stats(oracle.amplified(k));cache[key]=s;circuits.append(dict(states=states,stage=stage['stage'],threshold=threshold,k=k,**s))
                  if states==8 and k==1 and not any(x['stage']==stage['stage'] and x['k']==1 for x in validations):
                    tick=perf_counter();prob=float(Statevector.from_instruction(c).probabilities([oracle.flag])[1]);amp=float(table.probabilities@oracle.payload);expected=float(np.sin(3*np.arcsin(np.sqrt(amp)))**2);error=abs(prob-expected);assert error<1e-8
                    validations.append(dict(states=states,stage=stage['stage'],threshold=threshold,k=1,probability_error=error,seconds=perf_counter()-tick))
                s=cache[key];shots=rr['shots'];row=dict(run=run,states=states,alpha=alpha,method=method,seed=seed,query=query,stage=stage['stage'],threshold=threshold,round=rid,k=k,shots=shots,A_calls=shots*(2*k+1),weighted_cx=shots*s['cx'],weighted_u=shots*s['u'],weighted_depth=shots*s['depth'],**s);rows.append(row);runrows.append(row)
            assert sum(x['A_calls'] for x in runrows)==r['resources']['oracle_A_calls']
            results.append(dict(run=run,states=states,alpha=alpha,method=method,seed=seed,var_error_bp=abs(r['var']-truth['var'])*10000,es_error_bp=abs(r['es']-truth['es'])*10000,A_calls=sum(x['A_calls'] for x in runrows),weighted_cx=sum(x['weighted_cx'] for x in runrows),weighted_u=sum(x['weighted_u'] for x in runrows),weighted_depth=sum(x['weighted_depth'] for x in runrows),qubits=max(x['qubits'] for x in runrows),max_depth=max(x['depth'] for x in runrows)))
            print(f'Completed {run}',flush=True)
            pd.DataFrame(results).to_csv(OUT/'results.csv',index=False)
    for name,data in [('round_resources',rows),('circuits',circuits),('components',components),('quantization',quant)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    pd.DataFrame(results).groupby('states').agg(runs=('seed','size'),qubits=('qubits','max'),mean_A_calls=('A_calls','mean'),mean_weighted_cx=('weighted_cx','mean'),mean_weighted_u=('weighted_u','mean'),max_depth=('max_depth','max')).reset_index().to_csv(OUT/'summary.csv',index=False)
    write_json(OUT/'logs.json',logs);write_json(OUT/'validation.json',validations)
    write_json(OUT/'completion.json',dict(runs=len(results),rounds=len(rows),unique_circuits=len(circuits),validated_circuits=len(validations),max_probability_error=max(x['probability_error'] for x in validations),elapsed_seconds=perf_counter()-start,failures=0))
    write_json(OUT/'hashes.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='hashes.json'})
if __name__=='__main__':main()
