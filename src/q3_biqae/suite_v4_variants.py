"""Preregistered engineering posterior and one-factor stopping controls."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
import json
import numpy as np
import pandas as pd
from .config import ROOT, digest, write_json
from .suite_risk import load_inputs, configure, result_row, cp
from .exact_reference import risk
from .es_estimation import estimate_risk
from .amplitude import estimate_payload

BASE = ROOT/'artifacts/validation_suite_v4'
METHODS = ['iqae', 'grid_biqae', 'normal_biqae', 'beta_biqae']

def setup(name, protocol):
    dest = BASE/name
    dest.mkdir(parents=True, exist_ok=False)
    protocol.update(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_model='artifacts/feasibility_trained_v2/models/trained_risk_model.npz',
                    model_sha256=digest(ROOT/'artifacts/feasibility_trained_v2/models/trained_risk_model.npz'),
                    source_scenarios='artifacts/feasibility_trained_v2/generated_scenarios.npz',
                    weights='artifacts/feasibility_trained_v2/models/policy_lp.json',
                    source_hashes={p.name:digest(p) for p in (ROOT/'src/q3_biqae').glob('*.py')},
                    resource='expanded A/A-dagger calls, includes state preparation, analytic finite shots',
                    credibility='BIQAE intervals are Bayesian credible intervals, not proven frequentist confidence')
    write_json(dest/'protocol.json', protocol)
    return dest

def table_md(frame):
    cols = list(frame.columns)
    def fmt(x):
        return f'{x:.6g}' if isinstance(x, (float, np.floating)) else str(x)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None))

def aggregate(frame, keys, amplitude=False):
    rows=[]
    for key, g in frame.groupby(keys, dropna=False):
        key = key if isinstance(key,tuple) else (key,)
        row=dict(zip(keys,key)); n=len(g)
        row.update(n=n, exceptions=int((g.status!='ok').sum()))
        if amplitude:
            good=(g.error<=.005)&(g.status=='ok')
            row.update(mae=float(g.error.mean()), coverage=float(g.covered.fillna(False).mean()),
                       width=float(g.width.mean()))
        else:
            good=(g.var_error_bp<=5)&(g.es_error_bp<=5)&(g.status=='ok')
            row.update(var_mae_bp=float(g.var_error_bp.mean()), es_mae_bp=float(g.es_error_bp.mean()),
                       var_coverage=float(g.var_covered.fillna(False).mean()), es_coverage=float(g.es_covered.fillna(False).mean()),
                       es_width_bp=float(g.es_width_bp.mean()),
                       success_10bp=float(((g.var_error_bp<=10)&(g.es_error_bp<=10)&(g.status=='ok')).mean()),
                       var_resolved=float(g.var_resolved.fillna(False).mean()))
        lo,hi=cp(int(good.sum()),n)
        row.update(success_5bp_or_amp005=float(good.mean()),success_CP95_lower=lo,success_CP95_upper=hi,
                   mean_A_calls=float(g.A_calls.mean()), mean_shots=float(g.shots.mean()),
                   mean_unused_budget=float(g.unused_budget.mean()), mean_runtime_ms=float(g.runtime_ms.mean()))
        rows.append(row)
    return pd.DataFrame(rows)

def run_risks(dest, configurations, seeds):
    dist, raw, weights=load_inputs(); rows=[]; logs=[]
    for label,cfg,methods in configurations:
        for alpha in [.95,.975]:
            truth=risk(dist.losses(weights), dist.probabilities,alpha)
            for method in methods:
                for seed in seeds:
                    entry=dict(config=label, alpha=alpha, method=method, seed=seed, budget=65536)
                    try:
                        result=estimate_risk(dist,alpha,65536,seed,method,replace(cfg,alpha=alpha),weights)
                        entry.update(result_row(result,truth))
                        entry['unused_budget']=65536-entry['A_calls']
                        if entry['A_calls']>65536: raise AssertionError('budget exceeded')
                        logs.append(dict(config=label,alpha=alpha,method=method,seed=seed,logs=json.dumps(result['logs'])))
                    except Exception as exc:
                        entry.update(status='failed',error_message=repr(exc))
                    rows.append(entry)
                print(label,alpha,method,flush=True)
                pd.DataFrame(rows).to_csv(dest/'risk_results.csv',index=False)
    pd.DataFrame(logs).to_parquet(dest/'risk_rounds.parquet',index=False)
    df=pd.DataFrame(rows); summary=aggregate(df,['config','alpha','method'])
    summary.to_csv(dest/'risk_summary.csv',index=False)
    return df,summary

def finish(dest, started, counts):
    write_json(dest/'status.json',dict(status='complete',runtime_s=perf_counter()-started,**counts))
    write_json(dest/'checksums.json',{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='checksums.json'})

def variants():
    started=perf_counter(); cfg=configure()
    dest=setup('exp3_variants',dict(experiment=3, amplitudes=[.01,.025,.05,.3,.5,.9,.99],
          amplitude_seeds=list(range(100,130)),risk_seeds=list(range(100,200)),methods=METHODS,
          amplitude_epsilon=.005,risk_epsilon=.0002,budget=65536,common_config=cfg.to_dict(),
          algorithms={'normal':'after likelihood update, moment match theta to truncated normal on fixed mesh',
                      'beta':'after likelihood update, moment match amplitude to Beta density, include theta Jacobian',
                      'grid':'retain full discrete posterior',
                      'limitation':'engineering moment projections, not a reproduction of published Normal/Beta BIQAE schedules'},
          success='amplitude abs error <=.005; risk both VaR/ES <=5 or10bp; no hyperparameter selection'))
    rows=[]; logs=[]
    for amp in [.01,.025,.05,.3,.5,.9,.99]:
        for method in METHODS:
            for seed in range(100,130):
                row=dict(amplitude=amp,method=method,seed=seed)
                try:
                    t=perf_counter()
                    r=estimate_payload([1-amp,amp],[0.,1.],method,cfg,65536,seed,epsilon=.005)
                    row.update(status='ok',estimate=r.estimate,error=abs(r.estimate-amp),
                               covered=r.interval[0]<=amp<=r.interval[1],width=r.interval[1]-r.interval[0],
                               A_calls=r.ledger.oracle_A_calls,shots=r.ledger.shots_total,
                               unused_budget=65536-r.ledger.oracle_A_calls,runtime_ms=(perf_counter()-t)*1000,
                               interval_kind=r.interval_kind,converged=r.converged)
                    logs.append(dict(amplitude=amp,method=method,seed=seed,rounds=json.dumps(r.rounds)))
                except Exception as exc: row.update(status='failed',error_message=repr(exc))
                rows.append(row)
    adf=pd.DataFrame(rows); adf.to_csv(dest/'amplitude_results.csv',index=False)
    pd.DataFrame(logs).to_parquet(dest/'amplitude_rounds.parquet',index=False)
    aggregate(adf,['amplitude','method'],True).to_csv(dest/'amplitude_summary.csv',index=False)
    agg=aggregate(adf,['method'],True); agg.to_csv(dest/'amplitude_aggregate.csv',index=False)
    rdf,rs=run_risks(dest,[('common',cfg,METHODS)],range(100,200))
    report='# 实验3：网格、Normal与Beta后验工程版本正式对照\n\n'
    report+='同一已训练GMM、同一16状态冻结表与LP权重，重载并核对模型和生成样本。七个已知振幅各30种子，完整风险任务两个α各100种子；四方法共同预算65536、最大幂31、32轮、每轮256 shots、相同精度停止。\n\n'
    report+='## 算法边界\n\nNormal在每轮似然后将θ后验作正态矩匹配；Beta将振幅后验作Beta矩匹配，再用雅可比映回θ网格。两者在每一轮改变后验形状，并非仅修改初始Beta先验。四版本先验均为Beta(1,1)。这些是现有工程近似，仍依赖4097点网格计算矩，不能当作论文原版算法或声称节省网格计算。未改动既有后验公式。\n\n'
    report+='## 已知振幅\n\n'+table_md(agg[['method','n','mae','coverage','width','mean_A_calls','success_5bp_or_amp005']])+'\n\n'
    report+='## 训练后风险任务\n\n'+table_md(rs[['alpha','method','n','es_mae_bp','es_coverage','es_width_bp','mean_A_calls','success_5bp_or_amp005','success_10bp','success_CP95_lower']])+'\n\n'
    report+='误差相对于输入冻结表精确枚举，不包含分箱误差。Bayesian区间并无自动频率覆盖保证，覆盖率由重复种子实测。100/100成功时单配置双侧95% CP下界约96.38%，95/100约88.72%；未对多配置作优势显著性宣称。需同时阅读误差、覆盖率、区间宽度和查询量，低查询若伴随覆盖下降不构成优势。模型路径及源代码哈希见protocol.json；逐轮账本和失败保留。\n'
    (dest/'报告.md').write_text(report,encoding='utf-8')
    finish(dest,started,dict(amplitude_trials=len(adf),risk_trials=len(rdf),exceptions=int((adf.status!='ok').sum()+(rdf.status!='ok').sum())))

def controls():
    started=perf_counter();cfg=configure();methods=['iqae','grid_biqae']
    settings=[('baseline',cfg,methods),('maxk7',replace(cfg,max_grover_power=7),methods),
              ('maxk127',replace(cfg,max_grover_power=127),methods),
              ('var20',replace(cfg,var_budget_fraction=.2),methods),('var60',replace(cfg,var_budget_fraction=.6),methods),
              ('stop_interval256',replace(cfg,max_rounds=256),methods),
              ('stop_budget256',replace(cfg,max_rounds=256,stopping_rule='budget'),methods)]
    dest=setup('exp4_controls',dict(experiment=4,seeds=list(range(100,130)),alphas=[.95,.975],budget=65536,
               settings={label:c.to_dict() for label,c,_ in settings},methods=methods,
               design='one factor at a time; stopping pair shares 256 rounds; do not compare it to32-roundbaseline as stop-only',
               fixed_budget='disable precision early stop only; safety round limit, inconsistent IQAE interval, resolved VaR search still terminate'))
    rdf,rs=run_risks(dest,settings,range(100,130))
    report='# 实验4：Grover上限、预算分配与停止规则\n\n'
    report+='同一训练后GMM、冻结表和权重，α=95%/97.5%，每配置30种子，IQAE及网格BIQAE。共同总预算65536；每次只改一个因素。基准最大幂31、VaR/ES=40/60、区间精度停止、32轮。幂对照7/31/127，预算分配20/80、40/60、60/40。\n\n'
    report+='停止规则对照双方共同使用256轮（故其CP逐轮δ分配也相同），仅是否允许精度提前停止不同。budget模式仍受轮数与IQAE不一致安全退出限制；VaR确定后无需继续阈值搜索，因此整个风险任务可以有剩余预算。禁止将该剩余称作浪费或把固定预算模式伪称为必耗完总65536。\n\n'
    report+=table_md(rs[['config','alpha','method','es_mae_bp','var_resolved','es_coverage','es_width_bp','mean_A_calls','mean_unused_budget','success_5bp_or_amp005','success_10bp']])+'\n\n'
    report+='最大幂是允许上限，实际幂由区间单调分支选择；所有实际k、shots和A调用保存在risk_rounds.parquet。这里是理想有限测量采样，不是硬件速度。扩大幂可能增加电路深度；更高成功率若区间覆盖差亦不能作为可靠性优势。每配置30种子只能给探索性证据（30/30双侧95% CP下界约88.4%）。结果不根据测试数据调参再宣称泛化；该消融用于定位局限。误差仅相对冻结表，分箱偏差不会被改变这些设置解决。\n'
    (dest/'报告.md').write_text(report,encoding='utf-8')
    finish(dest,started,dict(risk_trials=len(rdf),exceptions=int((rdf.status!='ok').sum())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['variants','controls']);args=p.parse_args()
    globals()[args.experiment]()
