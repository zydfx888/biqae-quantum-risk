"""Merge measured v6 results, preserving the earlier results-only report."""
from pathlib import Path
import runpy
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'artifacts/validation_suite_v6'
runpy.run_path(str(ROOT/'scripts/update_report_v5.py'),run_name='__main__')
def rd(p): return pd.read_csv(B/p)
def tab(h,rows): return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
N={'grid_biqae':'网格BIQAE','paper_beta':'论文Beta','paper_normal':'论文Normal','paper_iqae_cp':'IQAE-CP','mc':'MC','qmc':'QMC','sobol':'QMC','is':'IS','iqae':'IQAE','normal_biqae':'工程Normal','beta_biqae':'工程Beta','dynamic_grid_biqae':'动态BIQAE','dynamic_mc':'动态MC','raw_exact':'原样本精确梯度','raw_lp':'原样本LP','equal':'等权'}
def name(v): return N.get(v,v)
p=['## 13. 论文Normal/Beta逐式实现结果',
'论文Algorithms 1–6的独立无网格实现完成5600次已知振幅和800次训练后风险估计，运行异常0次。Normal使用解析正态更新与Delta方法传播，Beta使用1000次后验抽样、变换和最大似然拟合。1000个分支反演检查最大残差1.49×10⁻¹⁴，正态积分均值差5.55×10⁻¹⁷，Beta拟合最大得分残差5.74×10⁻¹⁰；76310轮的调用核算与分支检查均通过。',
tab(['每批shots','版本','振幅MAE','区间覆盖率','0.005误差成功率','平均A/A†调用'],[[x.batch,name(x.method),f'{x.mae:.6f}',f'{x.coverage:.2%}',f'{x.success:.2%}',f'{x.mean_A_calls:,.0f}'] for x in rd('paper/amplitude_aggregate.csv').itertuples()]),
tab(['水平','版本','ES误差bp','ES区间宽bp','ES覆盖率','联合5bp成功数/100','平均A/A†调用'],[[f'{100*x.alpha:g}%',name(x.method),f'{x.es_mae_bp:.3f}',f'{x.mean_es_width_bp:.3f}',f'{x.es_coverage:.0%}',round(x.success_5bp*100),f'{x.mean_A_calls:,.0f}'] for x in rd('paper/risk_summary.csv').itertuples()]),
'**论文Beta在本次两个风险水平均100/100达到联合5bp目标；论文Normal存在明显失校，不能用更少调用宣称优势。** Normal边界零方差涉及433次试验，每批10 shots时振幅覆盖率仅41.29%；风险ES覆盖率为87%/49%。论文Beta的100/100成功对应单配置95%置信下界96.38%，不是跨任务保证。',
'该结果补齐了论文核心更新公式的独立实现；论文自身的正态传播及Beta抽样拟合仍是近似，没有证明精确贝叶斯等价。共享外部最大幂31和65536调用预算下，调度采用最大合法标准幂；与作者代码粗步长调度的审计出现121处分歧，因此**不是作者程序逐步一致复刻，也未复现论文无上限渐近复杂度结论**。IQAE-CP与论文两版本共享阶段调度，网格保留自身调度，表中比较的是完整算法。第10节的工程矩匹配版本与本节论文公式版本是不同实现。',
'## 14. 系统噪声结果']
n=rd('noise/summary.csv')
# Retain the first bounded implementation and show the independently rerun author scheduler.
p.pop()
p += ['### 13.1 作者原调度交叉验证',
'另完成5600次振幅与800次风险估计。直接使用作者原FindNextK调度方法后，10000个随机区间对照差异0处、非法分支0处；作者建议幂超过31时维持当前幂，预算仍在外部统一限制。',
tab(['水平','版本','ES误差bp','ES区间宽bp','ES覆盖率','联合5bp成功数/100','平均A/A†调用'],[[f'{100*x.alpha:g}%',name(x.method),f'{x.es_mae_bp:.3f}',f'{x.mean_es_width_bp:.3f}',f'{x.es_coverage:.0%}',round(x.success_5bp*100),f'{x.mean_A_calls:,.0f}'] for x in rd('paper_author_schedule/risk_summary.csv').itertuples()]),
tab(['每批shots','版本','振幅MAE','覆盖率','0.005成功率','平均调用'],[[x.batch,name(x.method),f'{x.mae:.6f}',f'{x.coverage:.2%}',f'{x.success:.2%}',f'{x.mean_A_calls:,.0f}'] for x in rd('paper_author_schedule/amplitude_aggregate.csv').itertuples()]),
'**消除调度实现差异后，论文Beta仍在两种风险水平均100/100通过联合5bp，ES误差0.562/0.686bp；Normal成功率仍为87%/50%、ES覆盖率87%/49%。主要结论未改变，Normal失校不能归因于先前调度差异。** Beta数值拟合器与作者不同，但求解同一最大似然目标；作者公开代码未提供Normal实现，Normal依据论文公式独立实现。该交叉验证不等于作者全部程序逐字复刻，也不证明近似后验与精确贝叶斯等价。公式与调度对应[论文v2](https://arxiv.org/html/2507.23074v2)及[作者公开实现](https://github.com/Kirin0570/BIQAE/blob/main/src/biae.py)。',
'## 14. 系统噪声结果']
p+=[tab(['状态','水平','门噪声λ','版本','ES误差bp','VaR/ES覆盖率','ES区间宽bp','平均CX执行量'],[[x.states,f'{x.alpha*100:g}%',f'{x.noise:g}',name(x.method),f'{x.es_mae_bp:.3f}',f'{x.var_coverage:.0%}/{x.es_coverage:.0%}',f'{x.es_width_bp:.2f}',f'{x.cx:,.0f}'] for x in n.itertuples()]),
'960次完整风险流程均已执行，覆盖8/16状态、两种风险水平、IQAE/网格BIQAE和四档门噪声，每格30次。理想精确参考的误差为0、无需采样区间；表中λ=0仍包含有限shots。Aer密度矩阵零噪声概率与理想计算的最大差为3.87×10⁻¹³，107个独特电路完成249次密度矩阵演化，随后按电路概率产生实际有限shots计数并驱动各轮自适应查询。',
'**16状态95%网格BIQAE的ES误差随λ从0升至10⁻⁴，由2.359bp升至18.368bp，ES覆盖率降为0；理想精度对门噪声不稳健。** 高噪声下100% ES覆盖来自约485.8bp的全支持宽区间，同时VaR覆盖为0，不能解释为可靠。8状态97.5%的零ES误差来自最大支持点与截断退化，也不是抗噪优势。该结果对应加载、Grover及反计算门上的退极化通道；不包含真实设备漂移、串扰或误差缓解，门数乘shots也不是真机耗时。',
'## 15. 动态分箱多窗口独立稳定性']
s=pd.DataFrame(json.loads((B/'rolling_results_conclusions.json').read_text(encoding='utf-8'))['stability_results'])
p+=[tab(['窗口','分箱','版本','分箱ES偏差bp','原样本LP差距bp','权重L2距离','真实测试ES'],[[x.window,'动态64' if x.representation=='dynamic64' else '原64',name(x.method),f'{x.es_bias_bp:+.3f}',f'{x.lp_gap_bp:.3f}',f'{x.lp_weights_l2:.3f}',f'{x.test_es:.4%}'] for x in s.itertuples()]),
'三个窗口共完成120个采样训练策略（每个窗口、分箱、方法各10个新种子）和3个原样本精确梯度策略。36个精确探针的最大VaR偏差4.558bp、ES偏差0.216bp、梯度相对误差0.00635。动态BIQAE平均ES分箱偏差为−0.106/−0.132/−0.150bp，说明分箱改善可跨已测窗口保持。',
'**分箱保真改善没有转化为稳定的样本外改善。** 动态BIQAE相对原分箱的真实测试ES变化为−1.712/+2.162/+8.074bp。窗口1和3的动态MC、BIQAE均各10/10达到原样本LP差距≤5bp及权重L2≤0.1；窗口2均0/10，BIQAE差距11.762bp、L2为0.270，原样本精确梯度的差距也为11.569bp。该窗口在已验证选择的步长与120步下尚未充分收敛，不能宣称动态分箱已普遍通过优化验收。',
'## 16. 年度滚动业务结果（2011—2025）']
b=rd('backtest/summary.csv')
p+=[tab(['策略','交易费bp/成交金额','年化收益','年化波动','日ES97.5%','最大回撤','年均换手率¹','累计费用/初始资金'],[[name(x.method),x.cost_bp,f'{x.annualized_return_252:.3%}',f'{x.annualized_volatility_252:.3%}',f'{x.daily_es_975:.3%}',f'{x.max_drawdown:.3%}',f'{x.mean_annual_turnover_excluding_initial:.3%}',f'{x.total_fee_initial_nav_units:.3%}'] for x in b.itertuples()]),
'15个年度模型完成45个GMM候选拟合、75次策略训练（包含验证候选）；5种策略×3种费率形成15条路径、56580条每日记录。训练、验证和年度测试的日期审计没有发现泄漏。¹换手率为目标权重与调仓前漂移权重之差的半L1范数，年均值排除初次入场；交易费对买入与卖出成交金额均收取。年内持仓漂移，权重约束用于年度调仓目标。',
'**10bp费用下，动态BIQAE年化13.041%、最大回撤−30.542%，原样本LP分别13.680%、−27.717%；未表现出相对LP的业务优势。** 动态BIQAE与MC的业务结果接近。每年度仅一个固定策略种子，这15条路径包含费率情景，不能当成15个独立市场样本或多种子置信认证；公开行业收益回测也不等于可交易成分股的实盘执行。']
daily=rd('backtest/daily_paths.csv');fig,ax=plt.subplots(figsize=(10,4.5))
for method,g in daily[daily.cost_bp_per_traded_dollar==10].groupby('method'):
    ax.plot(pd.to_datetime(g.date),g.nav,label=method,linewidth=1.25)
ax.set(ylabel='Net asset value (initial = 1)',title='Annual rolling backtest, 10 bp per traded dollar');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
fig.savefig(B/'annual_nav.png',dpi=160);plt.close(fig)
p+=['![年度滚动净值结果](artifacts/validation_suite_v6/annual_nav.png)','## 17. 尾部、资产维度及分布因子结果']
f=rd('factors/summary.csv');t=rd('factors/truths.csv')
key=f[(f.target_bp==5)&(f.states==64)]
agg=key.groupby(['alpha','assets','method']).agg(n=('n','sum'),es_mae_bp=('es_mae_bp','mean'),raw_es_mae_bp=('raw_es_mae_bp','mean'),success_rate=('success_rate','mean'),raw_success_rate=('raw_success_rate','mean'),es_coverage=('es_coverage','mean'),mean_queries=('mean_queries','mean')).reset_index()
p+=[tab(['水平','资产','版本','次数','离散ES误差bp','原样本ES误差bp','联合5bp成功率','原样本联合5bp率','ES覆盖率','平均查询'],[[f'{x.alpha*100:g}%',x.assets,name(x.method),x.n,f'{x.es_mae_bp:.2f}',f'{x.raw_es_mae_bp:.2f}',f'{x.success_rate:.1%}',f'{x.raw_success_rate:.1%}',f'{x.es_coverage:.1%}',f'{x.mean_queries:,.0f}'] for x in agg.itertuples()]),
'31个配置共24800次完整风险估计，异常0次；覆盖4/6/8资产、95%/97.5%/99%、正态相关性0.2/0.5/0.8、Student-t自由度3/5/10、非对称边际t-Copula、90/10压力混合及2%共损极端冲击。99%每个配置每方法100次，其他水平30次。上表按同资产数的9种64状态分布汇总，各分布等权；汇总成功率不是最差配置保证。',
'6/8资产采用训练后模型收益尺度扩展的合成因子分布，并非6/8个真实资产重新拟合的市场结果。离散参考和原样本参考均为同一有限生成样本的精确计算，不是连续总体尾部真值。经典查询计收益向量样本，量子查询计A/A†调用，二者不是相同硬件成本。']
families=key[key.alpha==.99].groupby(['kind','method']).agg(es_mae_bp=('es_mae_bp','mean'),raw_es_mae_bp=('raw_es_mae_bp','mean'),success_rate=('success_rate','mean'),raw_success_rate=('raw_success_rate','mean')).reset_index()
p +=[tab(['99%分布族','版本','离散ES误差bp','原样本ES误差bp','联合5bp成功率','原样本联合5bp率'],[[x.kind,name(x.method),f'{x.es_mae_bp:.2f}',f'{x.raw_es_mae_bp:.2f}',f'{x.success_rate:.1%}',f'{x.raw_success_rate:.1%}'] for x in families.itertuples()])]
factor_rows=[]
for x in t[(t.alpha==.99)&(t.states==64)&(t.assets==4)].itertuples():
    g=key[(key.id==x.id)&(key.alpha==.99)].set_index('method')
    factor_rows.append([x.kind,x.rho,x.df,f'{x.es_bias_bp:+.2f}',f"{g.loc['grid_biqae','success_rate']:.0%}",f"{g.loc['sobol','success_rate']:.0%}",f"{g.loc['grid_biqae','raw_success_rate']:.0%}",f"{g.loc['sobol','raw_success_rate']:.0%}"])
p += [tab(['4资产99%分布','ρ','ν','ES分箱偏差bp','网格5bp率','QMC 5bp率','网格原样本5bp率','QMC原样本5bp率'],factor_rows)]
st=t[(t.assets==4)&(t.kind=='normal')&(t.rho==.5)]
p +=[tab(['状态','水平','精确VaR分箱偏差bp','精确ES分箱偏差bp'],[[x.states,f'{100*x.alpha:g}%',f'{x.var_bias_bp:+.2f}',f'{x.es_bias_bp:+.2f}'] for x in st.sort_values(['states','alpha']).itertuples()]),
f'上述因子中，精确ES分箱偏差范围为{t.es_bias_bp.min():+.2f}至{t.es_bias_bp.max():+.2f}bp。**提高离散估计精度不能消除分箱对原样本尾部的偏差，99%尾部尤其不能只凭离散成功率验收。** 部分粗分箱使VaR与ES同时等于最大支持点，零超额载荷与支持裁剪可产生结构性零误差，不能作为量子估计优势。',
'### 17.1 振幅精度与置信参数结果']
a=rd('factors/amplitude_results.csv')
# Keep the detailed epsilon/delta factors visible without duplicating all amplitudes.
am=rd('factors/amplitude_summary.csv')
p +=[f'另完成4320次振幅估计，覆盖振幅0.01/0.05/0.5、误差目标0.05/0.02/0.01/0.005、δ=0.1/0.05/0.01及四个算法版本，每格30次。该精度单位是振幅概率，与ES的bp不同。']
p += [tab(['振幅目标ε','δ','版本','振幅MAE','覆盖率','达到ε比例','平均调用'],[[x.epsilon,x.delta,name(x.method),f'{x.mae:.6f}',f'{x.coverage:.1%}',f'{x.success:.1%}',f'{x.queries:,.0f}'] for x in am.itertuples()])]
cols=list(am.columns)
# Full machine factor summaries remain preserved; compact measured ranges in the report.
for method,g in am.groupby('method'):
    cov=next(c for c in ['coverage','coverage_rate'] if c in g)
    mae=next(c for c in ['mae','amplitude_mae'] if c in g)
    p.append(f'{name(method)}各配置振幅MAE范围{g[mae].min():.6f}—{g[mae].max():.6f}，实测区间覆盖率范围{g[cov].min():.1%}—{g[cov].max():.1%}。')
p+=['**已补齐分阶段单因素覆盖，未进行全部因素的笛卡尔积，也未证明跨配置的统一成功率。**', '## 18. 显式收益算术完整流程资源']
ar=rd('arithmetic/summary.csv')
p +=[tab(['状态','完整流程数','量子比特','平均A/A†调用','平均CX执行量','平均U执行量','最大深度'],[[x.states,x.runs,x.qubits,f'{x.mean_A_calls:,.0f}',f'{x.mean_weighted_cx:,.0f}',f'{x.mean_weighted_u:,.0f}',x.max_depth] for x in ar.itertuples()]),
tab(['状态','水平','量化VaR偏差bp','量化ES偏差bp'],[[x.states,f'{x.alpha*100:g}%',f'{x.var_bias_bp:+.2f}',f'{x.es_bias_bp:+.2f}'] for x in rd('arithmetic/quantization.csv').itertuples()]),
'24次完整CDF搜索与ES流程、495轮测量、54个独特编译电路已完成；所有8/16/32状态使用的20个基础A电路均经过实际状态向量验证，另7个8状态电路检查包含Grover幂1。工作寄存器归零，8状态概率最大偏差2.00×10⁻¹³，逆电路偏差低于3.46×10⁻¹³。',
'该电路已从组合损失查表扩展到资产收益qROM、受控整数权重乘加、显式损失寄存器、比较器、超额损失编码与反计算。四资产权重为[2,1,1,2]/6、各资产收益2位量化、损失寄存器5位；32状态需要14个量子比特，平均CX执行约6415万次，最大深度243452。',
'**当前是低位宽通用算术原型，ES量化偏差绝对值5.42—33.17bp，尚不满足5bp生产精度验收。** 资源结果包含已编译逻辑门与所有轮次shots的执行量，但不含设备路由、纠错和反馈延迟；也不能用不同任务预算的第12节查表数值直接计算算术开销倍率。达到目标精度的更高位宽资源成本仍未覆盖。',
'## 最终结论',
'**方案在理想模拟下的风险估计与优化流程可运行，动态分箱显著降低已测窗口的离散偏差，论文Beta核心公式实现也通过本次风险任务。现有结果不支持强经典基线优势、稳定的真实市场改善或端到端量子加速。**',
'论文Normal存在区间失校；动态分箱的第2窗口未通过优化验收；年度回测没有超越原样本LP；轻度门噪声已显著损害精度。99%尾部与更多资产的结果仍受分箱影响，通用算术的低位宽量化也未达到5bp要求。论文无上限复杂度、真实设备可靠性以及目标精度下的通用电路成本，尚未由这些结果证实。']
path=ROOT/'综合实验报告.md'
text=path.read_text(encoding='utf-8').split('## 最终结论')[0]
text=text.replace('动态分箱在已测策略中改善风险值、权重距离与最优性差距，但没有改善真实测试集表现，也未显示端到端加速。','动态分箱降低分箱偏差，但多窗口优化尚未普遍通过验收，真实测试表现没有稳定改善，也未显示端到端加速。')
path.write_text(text+'\n\n'+'\n\n'.join(p)+'\n',encoding='utf-8')
print('Merged all six v6 result areas')

