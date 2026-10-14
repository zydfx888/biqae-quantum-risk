"""Pre-registered full risk comparison on two trained-model-scaled stress laws."""
from time import perf_counter
import json
import numpy as np
import pandas as pd
from .config import ROOT, write_json, digest
from .trained_model import TrainedRiskModel, freeze_model_scenarios
from .exact_reference import risk
from .suite_risk import SOURCE, METHODS, initialize, finalize, configure, load_inputs, estimate, result_row, cp

OUT = ROOT/'artifacts/validation_suite_v4/exp2_distributions'

def generate_laws():
    model = TrainedRiskModel.load(SOURCE/'models/trained_risk_model.npz')
    mean_z = model.mixture @ model.means
    covariance_z = sum(p*(cov + np.outer(mu-mean_z, mu-mean_z))
                       for p, mu, cov in zip(model.mixture, model.means, model.covariances))
    mean = model.center + mean_z*model.scale
    cov = covariance_z*np.outer(model.scale, model.scale)
    rng = np.random.default_rng(2026091601)
    z = rng.multivariate_normal(np.zeros(len(mean)), cov, 20003)
    # df=4, standardized to retain the fitted GMM's unconditional covariance.
    student = mean + z*np.sqrt(2/rng.chisquare(4, 20003))[:, None]
    rng = np.random.default_rng(2026091602)
    stress = model.sample(20003, 2026091603)
    labels = rng.random(20003) < .10
    stress[labels] = (mean - 2*np.sqrt(np.diag(cov)) +
                      rng.multivariate_normal(np.zeros(len(mean)), 4*cov, labels.sum()))
    return {'student_t_df4':student, 'stress_mixture':stress}, labels, mean, cov

def summarize(data):
    rows = []
    for keys, g in data.groupby(['distribution','alpha','budget','method']):
        for target in [5,10]:
            hit = g.status.eq('ok') & g.var_error_bp.le(target) & g.es_error_bp.le(target)
            raw_hit = g.status.eq('ok') & g.raw_var_error_bp.le(target) & g.raw_es_error_bp.le(target)
            n, h = len(g), int(hit.sum())
            rows.append(dict(zip(['distribution','alpha','budget','method'],keys)) | {
                'target_bp':target, 'trials':n, 'failed_trials':int(g.status.ne('ok').sum()),
                'successes':h, 'success_rate':h/n, 'cp_lower':cp(h,n)[0], 'cp_upper':cp(h,n)[1],
                'simultaneous_cp_lower_per_distribution_alpha_target':cp(h,n,.05/15)[0],
                'global_cp_lower_120_config_targets':cp(h,n,.05/120)[0],
                'raw_empirical_success_rate':float(raw_hit.mean()),
                'success_and_narrow_ci_rate':float((hit & g.var_width_bp.le(2*target) & g.es_width_bp.le(2*target)).mean()),
                'var_mae_bp':g.var_error_bp.mean(), 'es_mae_bp':g.es_error_bp.mean(),
                'raw_var_mae_bp':g.raw_var_error_bp.mean(), 'raw_es_mae_bp':g.raw_es_error_bp.mean(),
                'var_coverage':g.var_covered.mean(), 'es_coverage':g.es_covered.mean(),
                'var_coverage_cp_lower':cp(int(g.var_covered.sum()),n)[0],
                'es_coverage_cp_lower':cp(int(g.es_covered.sum()),n)[0],
                'raw_var_coverage':g.raw_var_covered.mean(), 'raw_es_coverage':g.raw_es_covered.mean(),
                'var_width_bp':g.var_width_bp.mean(), 'es_width_bp':g.es_width_bp.mean(),
                'mean_queries':g.queries.mean(), 'p95_queries':g.queries.quantile(.95),
                'mean_shots':g.shots.mean(), 'mean_runtime_ms':g.runtime_ms.mean()})
    return pd.DataFrame(rows)

def report(directory, summary, matched, truths):
    lines = ['# 补充实验2：厚尾与压力混合分布完整风险估计', '',
        '## 结论边界', '',
        '两类分布均由已训练并重载的GMM提供位置、协方差或正常状态生成器；这是冻结的合成压力扩展，不是重新拟合的市场分布，也不把未知连续总体风险当作真值。',
        '所有方法在同一16状态表上完整运行VaR搜索和ES估计。另用同一批20003个原始收益向量精确枚举，揭示量子模块之前的压缩误差。', '',
        '## 预先固定的设置', '',
        '- Student-t：df=4，共享标量卡方尺度，多元相关结构和无条件协方差取自训练后GMM。',
        '- 压力混合：90%已训练GMM；10%均值下移各资产2个标准差、协方差放大4倍的高斯压力状态。',
        '- 每类20003个样本；原有等权组合排序尾部分箱，16状态；固定原LP权重；不按结果选权重或分布。',
        '- α=95%/97.5%；预算4096/16384/65536；5种方法；每配置100个种子100–199；目标5/10bp。',
        '- 配置、生成公式、种子、来源校验值先写入protocol.json，再生成数据和估计。', '',
        '## 原始样本与离散表的精确差异', '',
        '| 分布 | α | VaR偏差(bp) | ES偏差(bp) |', '|---|---:|---:|---:|']
    for r in truths.to_dict('records'):
        lines.append(f"| {r['distribution']} | {r['alpha']:.3f} | {r['var_bias_bp']:.3f} | {r['es_bias_bp']:.3f} |")
    lines += ['', '负数表示压缩后低估。下表误差目标针对离散表；不得解释为达到原始样本或连续总体的5/10bp精度。', '',
              '## 达到经验95%联合成功率的最小已测预算', '',
              '成功要求VaR和ES绝对误差同时达标。括号内是实际平均查询量；未达表示该预算网格内未满足。', '',
              '| 分布 | α | 目标bp | MC | QMC | IS | IQAE | BIQAE |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    observed = matched[matched.criterion.eq('observed_95pct')]
    for keys,g in observed.groupby(['distribution','alpha','target_bp']):
        cells = []
        for m in METHODS:
            r = g[g.method.eq(m)].iloc[0]
            cells.append(f'{int(r.min_budget)} ({r.mean_queries:.0f})' if r.reached else '未达')
        lines.append(f'| {keys[0]} | {keys[1]:.3f} | {keys[2]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 可靠性与资源解释', '',
        '区间覆盖率和平均宽度保存在summary.csv，逐次区间、查询、误差及原始样本参照保存在results.csv。BIQAE输出贝叶斯可信区间，其经验覆盖率需独立观察，不能自动等同95%频率学置信区间。',
        '经验95/100成功不等于以95%置信度证明成功率≥95%；单配置只有100/100时双侧CP下界约96.38%。另给出每分布/α/目标15配置Bonferroni界，以及全120配置-目标的同时界；100个种子不足以通过同时下界≥95%，不代表估计失败。',
        '查询口径为经典完整收益向量样本与量子的A/A†调用；QMC使用预先排序的已知离散分布逆CDF。排序、分布构造、量子加载电路和真实硬件代价不在此查询比中，因此不据此宣称端到端加速。',
        '原训练GMM的3000次参照见 validation_suite_v3/exp2_precision；本次不重复计数，也未覆盖旧结果。', '',
        '## 文件', '',
        '- protocol.json、environment.json：先验协议、依赖代码哈希。',
        '- generated_*.npz、frozen_*.npz：可重放的原始收益和冻结状态表。',
        '- exact_truths.csv、results.csv、rounds.parquet：精确参照、逐次结果、全部估计轮次。',
        '- summary.csv、matched_queries.csv：误差、区间、成本和成功率证据。',
        '- verification.json、status.json、checksums.json：验证、实际运行时间和文件哈希。']
    (directory/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    started = perf_counter()
    directory = initialize(OUT, {'experiment':'v4_extra_2', 'distributions':['student_t_df4','stress_mixture'],
        'generation':'Student t mean/cov from fitted GMM; df4 standardized covariance. Stress 90% fitted GMM, 10% Gaussian mean minus 2 marginal std, covariance times4.',
        'distribution_seeds':[2026091601,2026091602,2026091603], 'raw_samples':20003,
        'binning':'original freeze_model_scenarios with equal-weight ordering, 16 states',
        'alphas':[.95,.975], 'budgets':[4096,16384,65536], 'methods':METHODS,
        'seeds':list(range(100,200)), 'target_bp':[5,10], 'weights':'frozen original trained-model LP16 policy',
        'success':'VaR and ES absolute errors both <= target vs exact discrete enumeration',
        'reference_warning':'raw empirical enumeration is finite sampled reference, not continuous population truth',
        'multiplicity':'CP two-sided marginal; Bonferroni15 per distribution/alpha/target and120 overall',
        'stopping_config':configure().to_dict()})
    _, _, weights = load_inputs()
    laws, labels, mean, covariance = generate_laws()
    write_json(directory/'generation_parameters.json', {'mean':mean.tolist(), 'covariance':covariance.tolist(),
                'fixed_weights':weights.tolist(), 'realized_stress_count':int(labels.sum())})
    rows, logs, truths = [], [], []
    for name, raw in laws.items():
        np.savez_compressed(directory/f'generated_{name}.npz', returns=raw,
                            stress_labels=labels if name=='stress_mixture' else np.zeros(len(raw),dtype=bool))
        table = freeze_model_scenarios(raw,16)
        table.save(directory/f'frozen_{name}.npz')
        for alpha in [.95,.975]:
            truth = risk(table.losses(weights),table.probabilities,alpha)
            raw_truth = risk(-raw@weights,np.full(len(raw),1/len(raw)),alpha)
            truths.append({'distribution':name,'alpha':alpha,'var':truth['var'],'es':truth['es'],
                           'raw_var':raw_truth['var'],'raw_es':raw_truth['es'],
                           'var_bias_bp':(truth['var']-raw_truth['var'])*10000,
                           'es_bias_bp':(truth['es']-raw_truth['es'])*10000})
            for budget in [4096,16384,65536]:
                for method in METHODS:
                    print(f'{name} alpha={alpha} budget={budget} {method}: 100 seeds',flush=True)
                    for seed in range(100,200):
                        key = dict(distribution=name,alpha=alpha,budget=budget,method=method,seed=seed)
                        try:
                            r = estimate(table,weights,alpha,budget,seed,method,configure())
                            row = key | result_row(r,truth)
                            raw_row = result_row(r,raw_truth)
                            row.update({f'raw_{k}':raw_row[k] for k in ['var_error_bp','es_error_bp','var_covered','es_covered']})
                            assert row['queries']<=budget
                            logs.append(key | {'rounds_json':json.dumps(r['logs'],allow_nan=False)})
                        except Exception as exc:
                            row = key | dict(status='failed',error=repr(exc),var_covered=False,es_covered=False,
                                              raw_var_covered=False,raw_es_covered=False)
                        rows.append(row)
                    pd.DataFrame(rows).to_csv(directory/'results.csv',index=False)
    data = pd.DataFrame(rows)
    pd.DataFrame(logs).to_parquet(directory/'rounds.parquet',index=False,compression='zstd')
    truth_frame = pd.DataFrame(truths)
    truth_frame.to_csv(directory/'exact_truths.csv',index=False)
    summary = summarize(data)
    summary.to_csv(directory/'summary.csv',index=False)
    matches = []
    for keys,g in summary.groupby(['distribution','alpha','target_bp','method']):
        g = g.sort_values('budget')
        for criterion,column in [('observed_95pct','success_rate'),('marginal_CP95_lower_ge95pct','cp_lower'),
              ('simultaneous_CP95_lower_ge95pct','simultaneous_cp_lower_per_distribution_alpha_target'),
              ('global_CP95_lower_ge95pct','global_cp_lower_120_config_targets'),
              ('observed_and_narrow_CI95pct','success_and_narrow_ci_rate')]:
            selected = g[g[column]>=.95]
            r = selected.iloc[0] if len(selected) else None
            matches.append(dict(zip(['distribution','alpha','target_bp','method'],keys)) | {
                'criterion':criterion,'reached':r is not None,'min_budget':None if r is None else int(r.budget),
                'mean_queries':None if r is None else float(r.mean_queries),
                'successes':None if r is None else int(r.successes)})
    matched = pd.DataFrame(matches)
    matched.to_csv(directory/'matched_queries.csv',index=False)
    assert len(data)==6000 and data.groupby(['distribution','alpha','budget','method']).size().eq(100).all()
    assert data.status.eq('ok').all(), data[data.status.ne('ok')].to_dict('records')
    assert data.queries.le(data.budget).all()
    regenerated, regenerated_labels, _, _ = generate_laws()
    for name,raw in laws.items():
        np.testing.assert_array_equal(raw,regenerated[name])
    np.testing.assert_array_equal(labels,regenerated_labels)
    write_json(directory/'verification.json', {'trials':len(data),'groups':60,'seeds_per_group':100,
        'failed_trials':0,'all_queries_within_budget':True,'regeneration_exact_match':True,
        'original_gmm_reference_reused':str(ROOT/'artifacts/validation_suite_v3/exp2_precision'),
        'source_module_sha256':digest(__file__)})
    report(directory,summary,matched,truth_frame)
    finalize(directory,started,{'trials':len(data),'failed_trials':0,'distributions':2,'groups':60})
    print(directory,flush=True)

if __name__=='__main__':
    main()
