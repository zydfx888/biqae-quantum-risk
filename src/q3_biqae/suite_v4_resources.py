"""Full risk-flow circuit accounting, preserving the production estimators."""
from datetime import datetime, timezone
from time import perf_counter
import json
import traceback
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Statevector
from .config import ROOT, Config, digest, write_json
from .trained_model import TrainedRiskModel, freeze_model_scenarios
from .es_estimation import estimate_risk
from .tail_oracle import TailOracle, excess_payload
from .exact_reference import risk

OUT = ROOT/'artifacts/validation_suite_v4/exp5_resources'
SOURCE = ROOT/'artifacts/feasibility_trained_v2'

def main():
    start=perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Existing protocol; do not overwrite')
    cfg=Config(epsilon=.0002,budget=65536,max_grover_power=31,max_rounds=32,
               shots_per_round=256,grid_size=4097,record_posterior=False)
    tracked=[SOURCE/'models/trained_risk_model.npz', SOURCE/'generated_scenarios.npz',
             SOURCE/'models/policy_lp.json']+[ROOT/'src/q3_biqae'/f for f in
             ['suite_v4_resources.py','tail_oracle.py','measurement.py','var_search.py','es_estimation.py']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in tracked}
    write_json(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        states=[8,16,32],alphas=[.95,.975],methods=['iqae','grid_biqae'],seeds=[700,701,702],
        statevector_seeds=[700],config=cfg.to_dict(),source_hashes=hashes,
        execution='36 analytic finite-shot full risk runs; 12 independently executed full statevector finite-shot risk runs. Every logged threshold/payload/power compiled and statevector checked, cached across identical circuits.',
        compilation='u/cx basis, optimization_level=1, seed_transpiler=0; no hardware routing or gate duration model.',
        lookup='CDF and excess payload implemented as classically computed table-controlled rotations, not general arithmetic/comparison registers. Classical table construction timed separately.',
        cost='Per-round compiled gates and depth times shots; A forward k+1 and inverse k. Separately compiled loading/lookup blocks are descriptive and not additive after whole-circuit optimization.',
        failure='Retain exceptions, no seed replacement; accuracy reference is exact discrete table, not uncompressed population.'))
    model=TrainedRiskModel.load(SOURCE/'models/trained_risk_model.npz')
    with np.load(SOURCE/'generated_scenarios.npz') as f: raw=f['returns']
    assert np.array_equal(raw,model.sample(len(raw),20260915))
    weights=np.array(json.loads((SOURCE/'models/policy_lp.json').read_text())['weights'])
    results=[]; logs=[]; rows=[]; circuits=[]; blocks=[]; failures=[]; cache={}; blockcache={}

    def compile_stats(qc):
        tick=perf_counter()
        compiled=transpile(qc,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=0)
        return compiled,dict(compile_seconds=perf_counter()-tick,qubits=compiled.num_qubits,
            depth=compiled.depth(),cx=int(compiled.count_ops().get('cx',0)),u=int(compiled.count_ops().get('u',0)))

    for states in [8,16,32]:
        table=freeze_model_scenarios(raw,states); table.save(OUT/f'table_{states}.npz')
        losses=table.losses(weights)
        for alpha in [.95,.975]:
            truth=risk(losses,table.probabilities,alpha)
            for method in ['iqae','grid_biqae']:
                for seed,backend in [(s,'analytic_shots') for s in [700,701,702]]+[(700,'qiskit_statevector')]:
                    run_id=f'{states}_{alpha}_{method}_{seed}_{backend}'
                    tick=perf_counter()
                    try:
                        run_cfg=Config(**{**cfg.to_dict(),'backend':backend})
                        r=estimate_risk(table,alpha,cfg.budget,seed,method,run_cfg,weights)
                        runtime=perf_counter()-tick
                        logs.append(dict(run_id=run_id,logs=r['logs']))
                        run_rows=[]
                        for query,stage in enumerate(r['logs']):
                            tick=perf_counter()
                            payload=(losses<=stage['threshold']).astype(float) if stage['stage']=='var' else excess_payload(losses,stage['threshold'],float(np.ptp(losses)))
                            preprocessing=perf_counter()-tick
                            payload_key=(states,payload.tobytes())
                            if payload_key not in blockcache:
                                oracle=TailOracle(table.probabilities,payload)
                                lookup=QuantumCircuit(oracle.n+1)
                                for inst in oracle.circuit.data[len(oracle.preparation.data):]:
                                    lookup.append(inst.operation,inst.qubits,inst.clbits)
                                bid=f'payload_{len(blockcache):03d}'
                                validation=oracle.validate()
                                for name,qc in [('loading',oracle.preparation),('lookup',lookup),('A',oracle.circuit),('A_inverse',oracle.circuit.inverse())]:
                                    _,stats=compile_stats(qc)
                                    blocks.append(dict(payload_id=bid,states=states,block=name,**stats,**validation))
                                blockcache[payload_key]=(oracle,bid)
                            oracle,bid=blockcache[payload_key]
                            for rid,round_ in enumerate(stage['rounds']):
                                k=int(round_['k']); shots=int(round_['shots']); key=(bid,k)
                                if key not in cache:
                                    qc=oracle.amplified(k)
                                    compiled,stats=compile_stats(qc)
                                    tick=perf_counter()
                                    actual=float(Statevector.from_instruction(qc).probabilities([oracle.n])[1])
                                    svseconds=perf_counter()-tick
                                    amplitude=float(table.probabilities@payload)
                                    expected=float(np.sin((2*k+1)*np.arcsin(np.sqrt(np.clip(amplitude,0,1))))**2)
                                    stats.update(payload_id=bid,k=k,states=states,probability=actual,
                                        probability_error=abs(actual-expected),statevector_seconds=svseconds)
                                    assert stats['probability_error']<1e-8
                                    cache[key]=stats; circuits.append(stats)
                                    if k==0 or not any(x['states']==states and x['k']>0 for x in circuits[:-1]):
                                        with (OUT/f'{bid}_k{k}.qpy').open('wb') as f: qpy.dump(compiled,f)
                                stats=cache[key]
                                rr=dict(run_id=run_id,states=states,alpha=alpha,method=method,seed=seed,backend=backend,
                                    stage=stage['stage'],query=query,round=rid,threshold=stage['threshold'],
                                    payload_id=bid,k=k,shots=shots,successes=round_['successes'],
                                    preprocessing_seconds=preprocessing if rid==0 else 0,
                                    A_forward_calls=shots*(k+1),A_inverse_calls=shots*k,
                                    weighted_cx=shots*stats['cx'],weighted_u=shots*stats['u'],
                                    weighted_depth=shots*stats['depth'],**{f'circuit_{a}':stats[a] for a in ['depth','cx','u','qubits']})
                                rows.append(rr);run_rows.append(rr)
                        calls=sum(x['A_forward_calls']+x['A_inverse_calls'] for x in run_rows)
                        assert calls==r['resources']['oracle_A_calls']
                        if backend=='qiskit_statevector':
                            assert sum(x['weighted_cx'] for x in run_rows)==r['resources']['total_two_qubit_gates']
                        results.append(dict(run_id=run_id,states=states,alpha=alpha,method=method,seed=seed,backend=backend,
                            var=r['var'],es=r['es'],var_error_bp=abs(r['var']-truth['var'])*10000,
                            es_error_bp=abs(r['es']-truth['es'])*10000,var_resolved=r['var_resolved'],
                            runtime_seconds=runtime,rounds=len(run_rows),cdf_queries=sum(x['stage']=='var' for x in r['logs']),
                            A_calls=calls,shots=sum(x['shots'] for x in run_rows),
                            weighted_cx=sum(x['weighted_cx'] for x in run_rows),weighted_u=sum(x['weighted_u'] for x in run_rows),
                            weighted_depth=sum(x['weighted_depth'] for x in run_rows),max_depth=max(x['circuit_depth'] for x in run_rows),
                            qubits=max(x['circuit_qubits'] for x in run_rows)))
                    except Exception:
                        failures.append(dict(run_id=run_id,traceback=traceback.format_exc()))
                        write_json(OUT/'failures.json',failures)
                        print(f'FAILED {run_id}',flush=True)
                print(f'Finished states={states} alpha={alpha} method={method}',flush=True)
        pd.DataFrame(results).to_csv(OUT/'results_partial.csv',index=False)
    df=pd.DataFrame(results);df.to_csv(OUT/'results.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'round_resources.csv',index=False)
    pd.DataFrame(circuits).to_csv(OUT/'unique_circuits.csv',index=False)
    pd.DataFrame(blocks).to_csv(OUT/'component_resources.csv',index=False)
    write_json(OUT/'risk_logs.json',logs);write_json(OUT/'failures.json',failures)
    summary=df[df.backend=='analytic_shots'].groupby(['states','alpha','method']).agg(
        runs=('seed','size'),mean_A_calls=('A_calls','mean'),mean_cx=('weighted_cx','mean'),
        mean_u=('weighted_u','mean'),mean_shots=('shots','mean'),max_depth=('max_depth','max'),
        mean_weighted_depth=('weighted_depth','mean'),qubits=('qubits','max'),mean_ES_error_bp=('es_error_bp','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for (states,alpha,method),g in df[df.seed==700].groupby(['states','alpha','method']):
        if len(g)==2:
            paired.append(dict(states=int(states),alpha=float(alpha),method=method,
                var_difference=float(g['var'].max()-g['var'].min()),es_difference=float(g.es.max()-g.es.min()),
                query_difference=int(g.A_calls.max()-g.A_calls.min()),cx_difference=int(g.weighted_cx.max()-g.weighted_cx.min())))
    write_json(OUT/'paired_checks.json',paired)
    completion=dict(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=perf_counter()-start,
        runs=len(df),failures=len(failures),rounds=len(rows),unique_circuits=len(circuits),unique_payloads=len(blockcache),
        max_probability_error=max(x['probability_error'] for x in circuits),
        total_unique_compilation_seconds=sum(x['compile_seconds'] for x in circuits)+sum(x['compile_seconds'] for x in blocks),
        max_paired_ES_difference=max(x['es_difference'] for x in paired),
        source_hashes_unchanged=all(digest(ROOT/p)==h for p,h in hashes.items()))
    write_json(OUT/'completion.json',completion)
    lines=['# 实验5：完整风险流程电路资源核算','',
        '同一训练后 GMM、同一生成样本、固定 LP 权重。8/16/32 状态、95%/97.5%、IQAE/网格 BIQAE；每项3个种子，另对首个种子独立运行完整状态向量有限次测量流程。预算65536，最大 Grover 幂31。','',
        '## 结果','', '|状态|α|方法|A调用均值|CX执行次数均值|最大电路深度|量子比特|', '|---|---|---|---|---|---|---|']
    for r in summary.to_dict('records'):
        lines.append(f"|{r['states']}|{r['alpha']}|{r['method']}|{r['mean_A_calls']:.0f}|{r['mean_cx']:.0f}|{r['max_depth']}|{r['qubits']}|")
    lines.extend(['',f"实际完成 {len(df)} 次完整风险估计、{len(rows)} 个测量轮次、{len(circuits)} 个独特电路。失败 {len(failures)} 次。状态向量概率与解析值最大偏差 {completion['max_probability_error']:.3g}；配对 ES 最大差 {completion['max_paired_ES_difference']:.3g}。总墙钟 {completion['elapsed_seconds']:.2f} 秒。",'',
        '## 核算边界','',
        '- round_resources.csv 覆盖所有实际 CDF 阈值与 ES 超额损失载荷，列出每轮幂、shots、前向 A、逆 A、编译后 u/CX、深度及乘 shots 的执行量。重复电路缓存编译，但每一轮 shots 的执行成本重复计入。',
        '- component_resources.csv 分别编译概率加载、载荷查表、A 和 A 逆。完整电路使用 k 次 Grover，其中 A 前向 k+1 次、反计算 k 次。组件计数不与整体优化后的门数强行相加。',
        '- CDF 阈值与超额损失在经典端构成查表载荷，使用多控旋转；没有通用可逆收益计算/算术比较器。不得把本结果解读为其资源已全部实现。',
        '- 36 次解析有限采样流程均完整重放实际所需电路的编译与状态向量概率验证；另12次直接使用完整电路状态向量驱动全部自适应测量。每轮通过二项分布抽样 shots，未逐 shot 演化状态向量。',
        '- 门数乘 shots、深度乘 shots 是逻辑工作量，不是设备耗时。编译无耦合图约束；未包括路由、纠错、噪声、复位/读出/经典反馈延迟。模拟墙钟不能证明硬件加速。',
        '- 小样本只用于完整路径验证与资源曲线，不用于成功率统计认证。误差真值是各自离散分布，未消除分箱误差。','',
        '原始协议见 protocol.json；所有轮次 risk_logs.json；配对核查 paired_checks.json；QPY 保存每个载荷的 k=0 电路与每种状态规模的非零幂示例。'])
    (OUT/'实验5报告.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(completion),flush=True)

if __name__=='__main__': main()
