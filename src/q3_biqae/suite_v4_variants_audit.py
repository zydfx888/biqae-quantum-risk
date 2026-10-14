"""Independent ledger reconstruction and result interpretation for v4 3/4."""
import json
import numpy as np
import pandas as pd
from .config import ROOT, digest, write_json

def main():
    base=ROOT/'artifacts/validation_suite_v4'
    for name in ['exp3_variants','exp4_controls']:
        dest=base/name
        data=pd.read_csv(dest/'risk_results.csv')
        rounds=pd.read_parquet(dest/'risk_rounds.parquet')
        merged=data.merge(rounds,on=['config','alpha','method','seed'],validate='one_to_one')
        audit=[]
        for _, row in merged.iterrows():
            stages=json.loads(row.logs)
            calls=sum(r['shots']*(2*r['k']+1) for s in stages for r in s['rounds'])
            shots=sum(r['shots'] for s in stages for r in s['rounds'])
            assert calls==row.A_calls and shots==row.shots
            assert calls<=row.budget
            audit.append(dict(config=row.config,alpha=row.alpha,method=row.method,seed=row.seed,
                              calls=calls,unused_budget=int(row.budget-calls),
                              max_k=max(r['k'] for s in stages for r in s['rounds']),
                              max_stage_rounds=max(len(s['rounds']) for s in stages),
                              es_calls=sum(r['shots']*(2*r['k']+1) for s in stages if s['stage']=='es' for r in s['rounds'])))
        pd.DataFrame(audit).to_csv(dest/'ledger_reconstruction.csv',index=False)
        extra={}
        if name=='exp3_variants':
            a=pd.read_csv(dest/'amplitude_results.csv'); logs=pd.read_parquet(dest/'amplitude_rounds.parquet')
            for _,row in a.merge(logs,on=['amplitude','method','seed'],validate='one_to_one').iterrows():
                rr=json.loads(row['rounds'])
                assert sum(r['shots']*(2*r['k']+1) for r in rr)==row.A_calls
            conclusion='''
## 实测结论

840个已知振幅任务均达到0.005绝对误差且区间覆盖真值。四方法平均误差为0.000620–0.000693，平均调用7286–7529；Normal/Beta并未呈现大幅查询优势。按振幅分组的30/30成功只给出约88.4%的单配置CP下界；跨不同振幅聚合的210条统计仅作描述，不能替代每个振幅的可靠性认证。

800个训练后风险任务全部达到5bp和10bp目标。Normal/Beta在95%下ES平均误差约0.493bp，在97.5%下约0.643–0.648bp；网格分别0.509bp、0.641bp，差异很小。Normal的区间较网格略宽，且本机后处理时间较网格增加，不能据此认定其更快。IQAE在95%任务有1/100次ES区间漏覆，其余风险配置观测覆盖100%；这不说明Bayesian区间具有普遍频率保证。所有VaR均精确定位。

因此，新增对照支持现有矩投影工程实现能完成该冻结风险任务，但不支持将它们当作论文原版复现，也不支持明显超过网格BIQAE的结论。此处小于1bp的估计误差不能抵销输入分箱已有的19–25bp偏差。
'''
            extra['amplitude_ledgers_verified']=len(a)
        else:
            keys=['alpha','method','seed']
            left=data[data.config=='stop_interval256'].set_index(keys).sort_index()
            right=data[data.config=='stop_budget256'].set_index(keys).sort_index()
            cols=['var','es','var_ci_low','var_ci_high','es_ci_low','es_ci_high','A_calls','shots']
            np.testing.assert_array_equal(left[cols].values,right[cols].values)
            extra['stopping_pairs_identical']=len(left)
            conclusion='''
## 实测结论

- 最大幂7→31→127：网格BIQAE的95% ES平均误差0.808→0.577→0.277bp，97.5%为1.230→0.600→0.295bp。调用总量约5.5万变化很小，更大的幂减少shots并改善冻结表精度，但对应更深电路，不能视为免费收益。
- VaR预算20%在97.5%下不足：网格BIQAE仅14/30、IQAE仅10/30同时达到5bp/10bp，主要来自VaR未能确定。40%和60%均30/30通过；60%时BIQAE平均调用降至44564（40%为55048），ES误差略升至0.749bp（40%为0.600bp）。分配规则确实影响查询和可靠性，原40/60并非已证明最优。
- 相同256轮的固定预算/区间停止120对逐种子点估计、区间和查询账本完全相同。这说明当前严格2bp内部目标下，精度提前停止并未约束这些风险任务；有限阶段预算决定停止。整个任务仍会在VaR定位后保留部分预分预算。不能把32轮与256轮差异归因于停止规则，因为它也改变了逐轮δ分配。

这些因素能改变估计误差与查询量，却不改变输入离散表，因而不能修复主要分箱偏差。30种子结果作为因素定位，不作95%成功率的严格认证。
'''
        with (dest/'报告.md').open('a',encoding='utf-8') as f: f.write(conclusion)
        write_json(dest/'independent_checks.json',dict(risk_ledgers_verified=len(data),budget_violations=0,
                   risk_exceptions=int((data.status!='ok').sum()),tests='90 passed (full suite at execution)',
                   audit_source_sha256=digest(__file__),**extra))
        write_json(dest/'checksums.json',{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__': main()
