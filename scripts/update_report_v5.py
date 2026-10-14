"""Replace formal comparison sections with the expanded measured results."""
from pathlib import Path
import runpy
import re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/validation_suite_v5'
runpy.run_path(str(ROOT/'scripts/results_only_report.py'),run_name='__main__')
N={'iqae':'IQAE','grid_biqae':'网格BIQAE','normal_biqae':'Normal-BIQAE','beta_biqae':'Beta-BIQAE'}
C={'baseline':'幂31 / VaR40%','maxk7':'最大幂7','maxk127':'最大幂127','var20':'VaR20%','var60':'VaR60%'}
def rd(p):return pd.read_csv(BASE/p)
def tab(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
v=rd('variants/risk_summary.csv');amp=rd('variants/amplitude_aggregate.csv');paired=rd('variants/risk_paired.csv')
parts=['## 10. Normal/Beta与网格BIQAE正式对照（新增100种子结果）',
tab(['水平','预算','版本','ES平均误差bp','ES区间宽bp','ES覆盖率','5bp/10bp成功数','平均A/A†调用'],[
    [f'{x.alpha*100:g}%',f'{x.budget:,}',N[x.method],f'{x.es_mae_bp:.3f}',f'{x.mean_es_width_bp:.2f}',f'{x.es_coverage:.0%}',f'{round(x.success_5bp*x.n)}/{round(x.success_10bp*x.n)}（各{x.n}次）',f'{x.mean_A_calls:,.0f}'] for x in v.itertuples()]),
'2400次完整风险估计均完成，异常0次。65536预算下，两种风险水平的四版本均100/100达到5bp和10bp目标；Normal/Beta相对网格的ES误差配对区间均包含0。**高预算下精度相近，未发现Normal/Beta稳定优于网格版。**',
'4096预算下，95%水平网格/Normal/Beta联合5bp成功率为54%/46%/58%，97.5%为1%/0%/1%；16384预算的部分配置没有单调改善。低预算区间覆盖率虽为100%，平均ES区间仍宽约389–485bp。**宽区间覆盖不能当作高精度证据；有限VaR搜索预算仍限制完整风险任务。**',
tab(['水平','版本−网格','预算65536的ES误差差bp','95%配对区间bp'],[
    [f'{x.alpha*100:g}%',N[x.method],f'{x.mean_difference:+.4f}',f'[{x.bootstrap_low:+.4f}, {x.bootstrap_high:+.4f}]'] for x in paired[(paired.budget==65536)&(paired.metric=='es_error_bp')&paired.method.isin(['normal_biqae','beta_biqae'])].itertuples()]),
'上述区间为单项边际比较，不能作为全网格同时优势声明；100/100成功的单配置95%置信下界为96.38%，也不能自动扩展成所有配置的联合保证。',
tab(['已知振幅版本','次数','平均绝对误差','平均区间宽','覆盖次数','达到0.005误差次数','平均A/A†调用'],[
    [N[x.method],x.n,f'{x.mae:.7f}',f'{x.mean_width:.7f}',f'{round(x.coverage*x.n)}/{x.n}',f'{round(x.success*x.n)}/{x.n}',f'{x.mean_A_calls:,.0f}'] for x in amp.itertuples()]),
'2800次振幅估计均达到0.005绝对误差，每个版本覆盖699/700。Normal的平均后验投影差为8.54×10⁻⁶、最大1.40×10⁻⁴；Beta分别9.58×10⁻⁷和2.81×10⁻⁵，网格独立回放最大差2.82×10⁻¹⁴。**本次未发现明显投影失校，但不能据此保证其他振幅或预算同样可靠。**',
tab(['风险任务版本','非采样计算时间ms'],[[N[x.method],f'{x.non_sampling_ms:.3f}'] for x in rd('variants/timing_summary.csv').query("task == 'risk'").itertuples()]),
'Normal/Beta的非采样计算时间高于网格版，**没有降低后处理成本**。该时间包含初始化、调度及后验更新，非量子硬件时间。Normal/Beta为依赖网格的逐轮矩匹配工程版本，与仅更换先验不同；尚未验证为论文原算法等价实现。以上风险误差仅对离散表，不能抵销分箱偏差。']
c=rd('controls/summary.csv');c=c[c.target_bp==5]
parts+=['## 11. Grover上限与VaR/ES预算分配（新增100种子结果）',tab(['单因素配置','水平','版本','ES误差bp','ES区间宽bp','覆盖率','5bp成功数','平均A/A†调用'],[
    [C.get(x.config,x.config),f'{x.alpha*100:g}%',N[x.method],f'{x.es_mae_bp:.3f}',f'{x.es_width_bp:.2f}',f'{x.es_coverage:.0%}',f'{x.successes}/{x.n}',f'{x.mean_A_calls:,.0f}'] for x in c.itertuples()]),
'4000次完整风险消融均完成，异常0次。网格BIQAE最大幂7→31→127时，95%水平ES误差0.991→0.509→0.282bp，97.5%为1.195→0.641→0.290bp。**扩大幂上限改善理想估计精度，但电路更深，不是免费收益。**',
'VaR预算20%时，97.5%水平5bp成功率为IQAE41%、网格60%、Normal65%、Beta60%；40%与60%均100/100。网格60%时，平均调用降至41943/44145，而40%为53056/55572；ES误差增至0.592/0.784bp。**VaR预算不足会损害可靠性；60%在本任务中节省查询，但有限网格未证明全局最优。**',
'100/100的单配置置信下界为96.38%，40配置同时校正后仍不足95%；不能把逐项通过解释为整张网格已获得95%保证。']
s=rd('stopping/summary.csv');pairs=rd('stopping/paired.csv');rows=[]
for p in pairs.itertuples():
    group=s[(s.epsilon_bp==p.epsilon_bp)&(s.alpha==p.alpha)&(s.method==p.method)]
    a=group[group.stopping=='interval'].iloc[0];b=group[group.stopping=='budget'].iloc[0]
    rows.append([p.epsilon_bp,f'{100*p.alpha:g}%',N[p.method],f'{a.mean_A_calls:,.0f}/{b.mean_A_calls:,.0f}',f'{a.es_mae_bp:.3f}/{b.es_mae_bp:.3f}',f'{round(a.success_5bp_or_amp005*100)}/{round(b.success_5bp_or_amp005*100)}',f'{round(a.success_10bp*100)}/{round(b.success_10bp*100)}',f'{a.es_coverage:.0%}/{b.es_coverage:.0%}'])
parts+=['### 11.1 停止规则的精度—查询权衡（新增4800次）',tab(['内部目标bp','水平','版本','区间/预算停止A调用','区间/预算停止ES误差bp','5bp成功数（各100）','10bp成功数（各100）','ES区间覆盖率'],rows),
'同一内部目标的两种停止规则共用256轮上限；表内每格按“区间停止/预算停止”排列。内部目标指ES区间半宽目标，该精度数值还被复用于CDF概率容差，不表示VaR误差停止阈值。内部目标、最终5/10bp实测成功率不是同一个量。固定预算模式仍允许VaR定位后终止搜索，因此不一定耗尽总预算。']
for eps,g in pairs.groupby('epsilon_bp'):
    n=int(g.pairs.sum());few=int(g.fewer_query_pairs.sum());saving=-g.interval_minus_budget_queries.mean();delta=g.interval_minus_budget_es_abs_error_bp.mean()
    parts.append(f'内部目标{eps:g}bp：{few}/{n}组配对减少调用，平均减少{saving:,.0f}次A/A†，ES绝对误差平均变化{delta:+.3f}bp。')
parts.append('5bp内部目标下，各配置减少查询14.06%–26.08%，实测联合5bp均100/100；10bp内部目标下减少37.04%–54.68%，联合10bp均100/100，但部分联合5bp仅98/100或99/100，其95%成功率下界仅92.96%或94.55%。**停止规则的效果取决于目标精度是否在预算耗尽前达到。节省查询伴随精度权衡，不应单独当作算法优势；这些设置也不能修复输入分箱误差。**')
text=(ROOT/'综合实验报告.md').read_text(encoding='utf-8')
text=re.sub(r'## 10\. Normal/Beta工程版本结果[\s\S]*?(?=## 12\.)','\n\n'.join(parts)+'\n\n',text)
(ROOT/'综合实验报告.md').write_text(text,encoding='utf-8')
print('Expanded formal comparison and stopping results merged')
