"""Render measured results and conclusions from stored experiment summaries."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';V3=A/'validation_suite_v3';V4=A/'validation_suite_v4'
N={'mc':'MC','sobol':'QMC','is':'IS','iqae':'IQAE','grid_biqae':'BIQAE','exact':'精确梯度','lp':'LP','equal_weight':'等权','normal_biqae':'Normal-BIQAE','beta_biqae':'Beta-BIQAE'}
R={'original16':'原16状态','original64':'原64状态','static_actual64':'固定LP尾部64','adaptive_actual64':'当前权重动态64','raw_reference':'原始样本参考'}
def rd(p):return pd.read_csv(p)
def tab(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
parts=[]
def section(title,headers,rows,conclusion):parts.extend(['## '+title,tab(headers,rows),conclusion])
parts.append('# 方案三实验结果与结论\n\n**修正经典基线后，BIQAE接近精确参考，但没有优于强经典基线的证据。分箱权重错配是主要误差来源；动态分箱在已测策略中改善风险值、权重距离与最优性差距，但没有改善真实测试集表现，也未显示端到端加速。**\n\n风险误差单位bp：1bp=0.01个百分点。权重距离为欧氏距离。离散表误差、原始模型样本误差与真实测试风险分别列示。')
t=rd(V3/'exp1_training/summary.csv')
section('1. 修正经典基线后的训练结果（各30种子）',['方法','测试ES均值','种子标准差bp','距精确梯度权重距离','距LP权重距离'],[[N[x.method],f'{100*x.test_es_mean:.6f}%',f'{1e4*x.test_es_sd:.4f}' if x.n>1 else '—',f'{x.weights_l2_exact_mean:.5f}',f'{x.weights_l2_lp_mean:.5f}'] for x in t.itertuples()],
    'BIQAE相对精确梯度的测试ES均值高0.05702%；BIQAE−MC平均差+0.1524bp，95%配对区间[-0.7885,+1.1499]bp。**接近参考的结论保留，优于MC的结论不成立。** 区间反映训练种子随机性，不代表未来市场保证。')
w=rd(V3/'exp4_windows/all_summary.csv')
section('2. 三个时间窗口的测试结果',['方法','2011–2015测试ES','2016–2020测试ES','2021–2025测试ES'],[[N[m],*[f'{100*w[(w.window==i)&(w.method==m)].iloc[0].test_es_mean:.6f}%' for i in [1,2,3]]] for m in ['equal_weight','mc','sobol','is','iqae','grid_biqae','exact','lp']],
    'BIQAE相对精确梯度的ES均值差分别为−0.06367%、−0.01223%、+0.18405%。**接近参考的现象不限于原窗口，尚无跨市场优势证据。**')
m=rd(V3/'exp2_precision/matched_queries.csv');rows=[]
for alpha in [.95,.975]:
    for target in [5,10,20]:
        row=[f'{100*alpha:g}%',target]
        for name in ['mc','sobol','is','iqae','grid_biqae']:
            x=m[(m.alpha==alpha)&(m.target_bp==target)&(m.method==name)&(m.criterion=='marginal_CP95_lower_ge95pct')].iloc[0]
            row.append(f'{x.mean_queries_at_budget:,.0f}' if x.reached else '网格内未认证')
        rows.append(row)
section('3. 同误差查询量与可靠性（3000次）',['水平','目标bp','MC','QMC','IS','IQAE','BIQAE'],rows,
    '表内为单配置95%成功率置信下界达到95%时，最小已测预算对应的平均实际查询。97.5%水平5bp目标下，MC为99/100成功、下界94.55%；BIQAE为100/100、下界96.38%。全网格同时校正后，100/100下界仅93.80%，未取得全网格95%可靠性认证。**当前完整已知离散表任务中，QMC更省查询。**')
s=rd(V3/'exp2_precision/summary.csv')
parts.extend([tab(['水平','方法','预算65536的ES误差bp','VaR/ES区间宽bp','VaR/ES覆盖率'],[[f'{x.alpha*100:g}%',N[x.method],f'{x.es_mae_bp:.3f}',f'{x.var_width_bp:.3f}/{x.es_width_bp:.3f}',f'{x.var_coverage:.0%}/{x.es_coverage:.0%}'] for x in s[(s.target_bp==5)&(s.budget==65536)].itertuples()]),
    'QMC在4096预算也通过5bp点误差与区间半宽联合要求；MC/IS在已测预算内未通过该区间精度要求。经典样本与A/A†调用不是同等硬件成本；高覆盖率不等于区间足够窄。'])
d=rd(V4/'exp2_distributions/exact_truths.csv');s=rd(V4/'exp2_distributions/summary.csv')
section('4. 厚尾与压力分布结果（6000次）',['分布','水平','VaR分箱偏差bp','ES分箱偏差bp'],[['Student-t厚尾' if x.distribution=='student_t_df4' else '压力混合',f'{x.alpha*100:g}%',f'{x.var_bias_bp:.3f}',f'{x.es_bias_bp:.3f}'] for x in d.itertuples()],
    '两类分布、两个风险水平下，QMC在4096预算达到5bp离散精度目标，IS需16384，IQAE/BIQAE需65536。最高预算下，所有方法相对原始生成样本的5/10bp联合成功率均为0。**强经典基线更省查询、分箱主导总误差的结果在新增分布中仍成立。**')
parts.append(tab(['分布','水平','方法','预算65536的ES误差bp','ES覆盖率','ES区间宽bp'],[['厚尾' if x.distribution=='student_t_df4' else '压力',f'{x.alpha*100:g}%',N[x.method],f'{x.es_mae_bp:.3f}',f'{x.es_coverage:.0%}',f'{x.es_width_bp:.3f}'] for x in s[(s.budget==65536)&(s.target_bp==5)].itertuples()]))
d=rd(V3/'exp3_ablation/summary.csv')
section('5. VaR与状态数消融（900组）',['状态','水平','ES分箱偏差bp','阈值误差项bp','振幅/裁剪误差bp','估计→真值VaR的ES误差bp'],[[x.states,f'{x.alpha*100:g}%',f'{x.discretization_bp:.3f}',f'{x.threshold_bp:.3f}',f'{x.abs_sampling_amplitude_and_clip_bp:.3f}',f'{x.estimated_tau_es_abs_error_bp:.3f}→{x.true_tau_es_abs_error_bp:.3f}'] for x in d[d.method=='grid_biqae'].itertuples()],
    '**原分箱增加到64状态后，ES仍低估约19–21bp。** BIQAE真值VaR替换减少了部分搜索误差，无法消除输入压缩偏差。')
b=rd(V4/'exp1_binning/exact_summary.csv');rows=[]
for method,label in {'original_equal_tail90':'原分箱','uniform_mass_actual':'均匀等概率','uniform_width_actual':'均匀等损失宽','tail95_actual':'95%尾部加密','tail90_offgrid_actual':'90%尾部加密'}.items():
    for states in [16,32,64]:
        a=b[(b.method==method)&(b.states==states)&(b.scope=='anchor')].iloc[0];c=b[(b.method==method)&(b.states==states)&(b.scope=='cross_weight')].iloc[0]
        rows.append([label,states,f'{a.var_max_abs_bp:.2f}',f'{a.es_mae_bp:.3f}',f'{c.es_max_abs_bp:.2f}',f'{c.gradient_relative_l2_max*100:.2f}%',f'{c.centered_cosine_min:.3f}'])
section('6. 分箱方法与跨权重保真',['分箱','状态','锚点最大VaR误差bp','锚点平均ES误差bp','其他权重最大ES误差bp','最大梯度误差','最小方向余弦'],rows,
    '300条静态精确结果中，没有候选通过全部冻结权重检查。95%尾部加密的锚点ES接近零误差来自边界对齐的结构性保留；非对齐90%尾部64状态锚点ES最大误差0.172bp，其他权重最大误差却达74.79bp。**单组合分箱没有保留跨权重尾部结构。**')
parts.extend([tab(['当前权重动态64状态：20条精确探针','结果'],[['最大VaR误差','5.283bp'],['最大ES误差','0.211bp'],['最大尾概率误差','0.162个百分点'],['最大梯度相对误差','0.409%'],['最小中心化梯度方向余弦','0.999045']]),'**当前权重动态分箱通过有限探针精度门槛，不等于整个可行域都已验证。**'])
p=rd(V4/'exp1_binning/propagation/summary.csv')
section('7. 分箱改善后的重训结果（22个策略）',['表示','方法','原始模型ES','真实测试ES','最终分箱偏差bp','梯度相对误差'],[[R[x.representation],N[x.method],f'{100*x.raw_model_es:.6f}%',f'{100*x.test_es:.6f}%',f'{x.es_bias_bp:+.3f}',f'{100*x.gradient_relative_l2:.3f}%'] for x in p.itertuples()],
    '**固定LP分箱重训后重新出现约26–32bp偏差；动态分箱最终偏差约0.14bp。** 动态方案98个轨迹审计点最大ES误差0.162bp、最大VaR误差3.930bp、最小方向余弦0.99896。\n\n**真实测试ES未改善。** BIQAE原64状态测试ES为2.763194%，动态64状态为2.810247%。动态分箱约2.37秒/策略、BIQAE总训练约7.47秒；原始样本精确梯度训练约0.45秒，未显示端到端加速。')
o=rd(A/'optimization_acceptance/summary.csv');tr=rd(A/'optimization_acceptance/trajectories.csv');refs=json.loads((A/'optimization_acceptance/lp_references.json').read_text())
section('8. 权重、最优性差距与优化验收（新增核算）',['LP参考','Cnsmr','Manuf','HiTec','Hlth','最小模型ES'],[[label,*[f'{v:.2%}' for v in refs[k]['weights']],f'{100*refs[k]["objective"]:.6f}%'] for k,label in [('frozen16','原16状态'),('raw20003','原始20003样本')]],
    '**压缩改变了最优权重：原16状态LP为[0.4,0,0,0.6]，原始样本LP为[0.6,0,0,0.4]。** 仅接近离散LP不能证明原模型优化可靠。')
parts.extend([tab(['表示','方法','策略数','平均/最大目标差bp','平均/最大LP权重距离','联合通过数'],[[R[x.representation],N[x.method],x.n,f'{x.mean_gap_bp:.3f}/{x.max_gap_bp:.3f}',f'{x.mean_weights_l2_lp:.4f}/{x.max_weights_l2_lp:.4f}',f'{round(x.joint_pass_rate*x.n)}/{x.n}'] for x in o.itertuples()]),
    '联合结果对应新增工程验收线：模型目标差≤1bp、权重距离≤0.05、约束误差≤10⁻⁸。原16状态行参照离散表LP，其余行参照原始20003样本LP；不等于LP唯一性或市场可靠性认证。',
    '**风险值接近不保证权重接近。** 原16状态BIQAE平均目标差0.595bp，平均LP权重距离0.1085，仅7/30联合通过；动态64状态BIQAE平均目标差0.356bp、权重距离0.0232，最大分别0.587bp和0.0351，3/3通过；动态MC同样3/3通过。小样本支持优化传导改善，不支持BIQAE优于MC。'])
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for rep,label,color in [('raw_reference','Raw exact','#6b7280'),('original64','Original64 BIQAE','#a98a5a'),('static_actual64','Static LP64 BIQAE','#d46f5e'),('adaptive_actual64','Adaptive64 BIQAE','#148c81')]:
    q=tr[(tr.dataset=='v4_raw20003')&(tr.representation==rep)&(tr.method==('exact' if rep=='raw_reference' else 'grid_biqae'))].groupby('step')[['gap_bp','weights_l2_lp']].mean()
    axes[0].plot(q.index,q.gap_bp,label=label,color=color);axes[1].plot(q.index,q.weights_l2_lp,label=label,color=color)
axes[0].set(xlabel='Training step',ylabel='Objective gap to raw-model LP (bp)',title='Saved training convergence');axes[1].set(xlabel='Training step',ylabel='L2 distance to raw-model LP',title='Weight convergence')
for ax in axes:ax.legend(fontsize=8,frameon=False)
image=A/'optimization_acceptance/convergence.png';fig.savefig(image,dpi=180);plt.close(fig)
parts.extend([f'![目标差距与权重距离收敛结果]({image.as_posix()})',tab(['表示/方法','首步平均差距bp','第120步差距bp','末20步平均波动bp'],[[R[x.representation]+'/'+N[x.method],f'{x.mean_first_gap_bp:.3f}',f'{x.mean_gap_bp:.3f}',f'{x.mean_last20_gap_range_bp:.3f}'] for x in o[(o.dataset=='v4_raw20003')&o.method.isin(['grid_biqae','exact'])].itertuples()]),
    '动态BIQAE平均差距由6.530bp降至0.356bp，末20步平均波动范围0.052bp；固定LP分箱仍为17.046bp。**动态方案在已记录轨迹中同时接近原模型目标和权重参考，不据有限120步结果宣称渐近收敛。**'])
q=rd(V3/'exp5_quantum/aggregate.csv')
section('9. 量子实现、网格与先验结果',['方法','网格','先验','次数','振幅平均误差','覆盖率','平均A调用'],[[N.get(x.method,x.method),x.grid,x.prior,x.runs,f'{x.mae:.7f}',f'{x.coverage:.0%}',f'{x.mean_A_calls:,.0f}'] for x in q[q.section.isin(['known','official'])].itertuples()],
    '1470次已知振幅、21次官方IQAE、24次完整电路与24次解析对照完成；4283轮未发现象限错选，最大电路概率偏差3.14×10⁻¹²，24组配对一致。**未发现明显实现偏差。** 网格加密未明显改善平均误差，错配先验使部分误差上升；官方IQAE停止规则不同，其调用量不构成公平优势比较。')
v=rd(V4/'exp3_variants/risk_summary.csv')
section('10. Normal/Beta工程版本结果',['水平','方法','次数','ES误差bp','区间宽bp','覆盖率','平均A调用','5bp成功率'],[[f'{x.alpha*100:g}%',N[x.method],x.n,f'{x.es_mae_bp:.3f}',f'{x.es_width_bp:.3f}',f'{x.es_coverage:.0%}',f'{x.mean_A_calls:,.0f}',f'{x.success_5bp_or_amp005:.0%}'] for x in v.itertuples()],
    '840个已知振幅任务均达到0.005误差并覆盖真值；800个风险任务均达到5/10bp目标。**Normal/Beta误差约0.49–0.65bp，与网格差别很小，没有明显优势。** 两者是逐轮矩投影工程版本，尚非已验证等价的论文原算法。')
c=rd(V4/'exp4_controls/risk_summary.csv')
section('11. Grover、预算分配与停止结果',['配置','水平','方法','ES误差bp','ES覆盖率','平均A调用','5bp成功数'],[[x.config,f'{x.alpha*100:g}%',N[x.method],f'{x.es_mae_bp:.3f}',f'{x.es_coverage:.1%}',f'{x.mean_A_calls:,.0f}',f'{round(x.success_5bp_or_amp005*x.n)}/{x.n}'] for x in c.itertuples()],
    '最大幂31→127使BIQAE误差约减半，但增加电路深度；97.5%水平VaR预算20%时BIQAE仅14/30成功，40%/60%均30/30。60%预算时平均调用约44564，低于40%时55048，ES误差由0.600升至0.749bp。共同256轮的固定预算/区间停止120组配对结果一致。**参数设置影响查询和精度，不能修复分箱偏差。**')
r=rd(V4/'exp5_resources/summary.csv')
section('12. 完整风险流程资源（含32状态）',['状态','水平','方法','平均A调用','平均CX执行量','最大深度','量子比特','ES误差bp'],[[x.states,f'{100*x.alpha:g}%',N[x.method],f'{x.mean_A_calls:,.0f}',f'{x.mean_cx:,.0f}',x.max_depth,x.qubits,f'{x.mean_ES_error_bp:.3f}'] for x in r.itertuples()],
    '48次完整流程、1513轮测量、25种载荷、192个独特电路完成，失败0次；12组解析/完整电路阈值、幂、shots、计数和CX成本一致，最大概率偏差6.31×10⁻¹²。\n\n**32状态每次完整流程各配置平均CX执行量约3868万–4067万，最大深度80308。** 包含CDF与ES的加载、查表、反计算和Grover；逻辑门数乘shots不是真机时间，未涵盖通用收益算术、路由、纠错及反馈延迟。8状态97.5%零载荷加裁剪产生零ES误差，但ES仍计入188928次CX；该退化不是精度优势。')
parts.append('## 最终结论\n\n**分箱方法、第二类分布与32状态资源均有实际运行结果；新增优化核算确认动态分箱改善了已测策略的权重和目标差距，而静态单组合分箱无法维持训练中的保真。现有结果仍不支持真实市场泛化改善、强经典基线优势或端到端量子加速。**')
(ROOT/'综合实验报告.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
print('Results-only report written')
