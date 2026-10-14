"""Render the five completed experiments from saved measurements; audit artifacts."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT, digest, write_json

BASE = ROOT/'artifacts/validation_suite_v3'
NAMES = {'mc':'MC','sobol':'QMC','is':'IS','iqae':'IQAE','grid_biqae':'BIQAE','exact':'精确梯度','lp':'LP','equal_weight':'等权'}

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])

def link(path, label):
    return f'[{label}]({(BASE/path).resolve().as_posix()})'

def run():
    audited = 0
    for manifest in BASE.rglob('checksums.json'):
        for name, value in json.loads(manifest.read_text(encoding='utf-8')).items():
            assert digest(manifest.parent/name) == value, (manifest, name)
            audited += 1
    p = BASE/'exp2_precision'
    # Cap the two-outcome Bonferroni p value at 1. Holm already did this.
    # This is reporting-only; point estimates, intervals, counts and decisions are unchanged.
    summary = pd.read_csv(p/'summary.csv')
    if (summary.coverage_under95_p>1).any():
        before = digest(p/'summary.csv')
        summary['coverage_under95_p'] = summary.coverage_under95_p.clip(upper=1)
        summary.to_csv(p/'summary.csv',index=False)
        write_json(p/'postprocessing.json', {
            'utc':datetime.now(timezone.utc).isoformat(),
            'change':'Cap reported two-outcome Bonferroni p values at 1; Holm p values and every decision already unchanged.',
            'previous_summary_sha256':before, 'new_summary_sha256':digest(p/'summary.csv'),
            'source_after_reporting_fix_sha256':digest(ROOT/'src/q3_biqae/suite_risk.py'),
            'raw_trials_unchanged':True})
    matched = pd.read_csv(p/'matched_queries.csv')
    a = pd.read_csv(BASE/'exp3_ablation/summary.csv')
    d = pd.read_csv(BASE/'exp3_ablation/discretization.csv')
    t = pd.read_csv(BASE/'exp1_training/summary.csv')
    w = pd.read_csv(BASE/'exp4_windows/all_summary.csv')
    r2 = pd.read_csv(p/'results.csv')
    r3 = pd.read_csv(BASE/'exp3_ablation/results.csv')
    assert len(r2)==3000 and len(r3)==900
    assert (r2.status=='ok').all() and (r3.status=='ok').all()
    assert r2.groupby(['alpha','budget','method']).size().eq(100).all()
    assert r3.groupby(['states','alpha','method']).size().eq(30).all()
    assert (r2.queries<=r2.budget).all() and (r3.full_queries<=65536).all()
    residual = np.abs(r3.discretization_bp+r3.threshold_bp+r3.sampling_amplitude_and_clip_bp-r3.total_vs_continuous_bp).max()
    assert residual < 1e-8
    rows=[]
    for alpha in [.95,.975]:
        for target in [5,10]:
            row=[f'{alpha*100:g}%',str(target)]
            for method in ['mc','sobol','is','iqae','grid_biqae']:
                z=matched[(matched.alpha==alpha)&(matched.target_bp==target)&(matched.method==method)&(matched.criterion=='marginal_CP95_lower_ge95pct')].iloc[0]
                row.append(f'{z.mean_queries_at_budget:,.0f} / {z.min_tested_budget:,.0f}' if z.reached else '网格内未认证')
            rows.append(row)
    query_table=table(['风险水平','目标bp','MC','QMC','IS','IQAE','BIQAE'],rows)
    coverage_table=table(['水平','方法','VaR/ES覆盖','VaR区间宽bp','ES区间宽bp','ES平均绝对误差bp'],[
        [f'{x.alpha*100:g}%',NAMES[x.method],f'{x.var_coverage:.0%} / {x.es_coverage:.0%}',f'{x.var_width_bp:.3f}',f'{x.es_width_bp:.3f}',f'{x.es_mae_bp:.3f}']
        for x in summary[(summary.target_bp==5)&(summary.budget==65536)].itertuples()])
    exp2=f'''# 实验2：相同误差、查询量与可靠性

## 结论

在当前已训练模型的16状态离散分布上，QMC在两种风险水平、5/10bp目标均以4096次向量采样达标，查询量明显低于BIQAE。BIQAE在5bp目标相对朴素MC有一定查询节省或更高观测成功率，但没有胜过QMC或IS。

固定已保存GMM与LP权重；α为95%和97.5%，保留4096/16384/65536预算网格，5种算法，每配置100种子，合计3000次，失败0次。真实模型训练来自历史训练集；本实验重复加载同一模型，未为估计方法重新拟合模型。每次成功定义为VaR和ES两个绝对误差都不超过目标，1bp=0.0001。

## 等误差比较

下表按“单配置双侧95% Clopper–Pearson成功率下界≥95%”选最小已测预算；单元格为**平均实际查询 / 预算上限**。它是网格内的边际比较，不是连续预算最优点或全网格联合保证。

{query_table}

α=97.5%、5bp下，MC在65536预算为99/100成功，观测率99%，但置信下界94.55%，所以表中没有认证；BIQAE为100/100，下界96.38%。99/100不能按“至少95%成功率已得到95%置信认证”解释。原20bp目标及按观测95%筛选的全部结果仍保留于matched_queries.csv。

15个方法×预算配置同时校正后，即使100/100成功，下界也仅93.80%，没有配置取得全网格95%可靠性认证。这反映100种子对该更强声明的证据不足，不等于精度不合格。不同alpha/目标也没有全局同时保证。

## 返回区间

以下为预算65536的平均宽度和观测覆盖，均针对离散真值：

{coverage_table}

全部预算配置的VaR覆盖100%，ES覆盖99%–100%；覆盖检验未发现显著低于95%的证据，也不能据此宣称所有配置都有95%频率保证。BIQAE区间是贝叶斯可信区间。

若还要求返回的两个区间半宽均≤目标且点误差达标，QMC在4096预算、两水平5bp目标均100/100通过；BIQAE和IQAE需65536预算。MC/IS在已测预算中未达这一要求，其保守区间较宽。QMC在4096预算的ES宽度分别4.744/9.488bp。

## 计费与边界

- 经典方法一次完整收益向量复用VaR/ES；量子计费为Σshots×(2k+1)，包括A与A†。这是抽象查询单位，不是同等硬件耗时。
- 当前QMC用已知完整离散分布按损失排序后做一维逆CDF采样，恰适合该小型查表问题；排序和表构建未进入查询数，不能外推到未知高维连续分布。
- 完整16状态分布已可经典枚举，精确参考本身就是很强的实际基线。本实验不证明相对精确枚举的量子优势。
- 训练与风险估计大量使用解析概率的有限shots模拟，秒级运行不代表真实量子电路速度。
- 执行前修正经典VaR区间上界：CDF带饱和时使用已知最高损失支持点，防止漏采尾部时虚假窄区间。新增回归测试。报告阶段仅将Bonferroni原始p值截到1；原始估计、区间和结论不变，记录见postprocessing.json。

原始结果：{link('exp2_precision/results.csv','3000次结果')}；查询筛选：{link('exp2_precision/matched_queries.csv','全部目标及四种认证口径')}；配置、逐轮日志和哈希均在同目录。实际运行20.49秒，低于60分钟上限。
'''
    ablation_table=table(['状态','水平','分箱ES偏差bp','BIQAE阈值项bp','振幅/裁剪绝对误差bp','估计VaR→真值VaR的ES误差bp'],[
        [x.states,f'{x.alpha*100:g}%',f'{x.discretization_bp:.3f}',f'{x.threshold_bp:.3f}',f'{x.abs_sampling_amplitude_and_clip_bp:.3f}',f'{x.estimated_tau_es_abs_error_bp:.3f} → {x.true_tau_es_abs_error_bp:.3f}']
        for x in a[a.method=='grid_biqae'].itertuples()])
    exp3=f'''# 实验3：VaR误差与16/32/64状态离散化消融

## 结论

当前方案的主要误差来自分箱压缩。状态数从16增到64，ES仍低估约19–21bp；提高振幅精度不能补偿该偏差。BIQAE在32/64状态存在VaR搜索误差，但其对ES的影响小于分箱项。

同一已训练GMM、完全相同的20003条生成收益、同一LP16权重；每状态×水平×方法30种子，5方法、两风险水平、每次65536预算，共900组，正式失败0次。未按评估权重重新分箱；沿用原来按等权损失排序再取各箱条件均值的压缩方法。

{ablation_table}

表中ES误差为相对该状态数的精确离散ES的平均绝对误差。真值VaR分支使用**该离散分布的精确VaR**，用来隔离搜索误差；原始20003样本的VaR/ES另外精确枚举，供分箱比较。这里的“原始真值”是同一模型生成样本的有限样本参考，不是未知总体真值，也不是市场测试集真值。

16状态下VaR已精确定位，因此替换真值VaR没有改善；32状态97.5%下，BIQAE的ES误差从1.266bp降到0.722bp，但分箱低估仍为22.540bp。64状态VaR误差也未单调消失，说明增加箱数不能解决排序权重与评估权重不一致导致的信息损失。

## 可核验误差分解

对每次结果使用带符号的严格恒等式：

`最终ES − 原始样本ES = (离散ES − 原始样本ES) + (离散RU目标在估计VaR处的值 − 离散ES) + (最终ES − 离散RU目标在估计VaR处的值)`

三项依次为分箱、阈值、采样/振幅与最终裁剪。900组最大残差为{residual:.3g}bp。绝对误差不能简单相加，报告未把互相抵消的误差误称为精度改进。

经典真值VaR分支复用同一批样本；量子分支复用相同ES阶段预算39322及随机种子。该分支有真值信息，仅为诊断，不能按其查询数当成可部署算法进行公平排序。全部方法及原始有符号分解见{link('exp3_ablation/results.csv','逐次结果')}、{link('exp3_ablation/summary.csv','汇总')}。

本次可行性验证支持算法在离散代理上的执行正确性；尚不支持对原模型尾部风险达到5/10bp总误差。后续应先改善面向实际组合损失的尾部压缩，并重新做训练/测试隔离验证，再增加量子查询。运行11.45秒，低于35分钟上限。
'''
    (p/'报告.md').write_text(exp2,encoding='utf-8')
    (BASE/'exp3_ablation/报告.md').write_text(exp3,encoding='utf-8')
    training_table=table(['方法','30种子测试ES均值','种子标准差bp'],[[NAMES[x.method],f'{100*x.test_es_mean:.6f}%',f'{10000*x.test_es_sd:.4f}' if x.n>1 else '—'] for x in t.itertuples()])
    windows_table=table(['测试窗口','MC测试ES','BIQAE测试ES','精确梯度测试ES','BIQAE相对参考差'],[
        [period,*[f'{100*w[(w.window==i)&(w.method==m)].iloc[0].test_es_mean:.6f}%' for m in ['mc','grid_biqae','exact']],f'{w[(w.window==i)&(w.method=="grid_biqae")].iloc[0].test_gap_exact_relative_pct:+.5f}%']
        for i,period in enumerate(['2011–2015','2016–2020','2021–2025'],1)])
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    z=matched[(matched.alpha==.975)&(matched.target_bp==5)&(matched.criterion=='observed_95pct')].set_index('method').loc[['mc','sobol','is','iqae','grid_biqae']]
    bars=axes[0].bar([NAMES[m] for m in z.index],z.mean_queries_at_budget,color=['#667085','#14866d','#5ab49c','#a19acb','#5544a5'])
    axes[0].bar_label(bars,labels=[f'{x:,.0f}' for x in z.mean_queries_at_budget],padding=3,fontsize=9)
    axes[0].set(title='97.5% risk, 5 bp: observed success >=95%',ylabel='Mean queries at smallest tested budget',ylim=(0,76000))
    axes[0].text(.5,.88,'MC: 99/100; others: 100/100',transform=axes[0].transAxes,ha='center',fontsize=9)
    q=a[(a.alpha==.975)&(a.method=='grid_biqae')].sort_values('states')
    xx=np.arange(3);width=.25
    for k,(col,label,color) in enumerate([('abs_discretization_bp','Discretization','#db8842'),('abs_threshold_bp','VaR search','#598fb5'),('abs_sampling_amplitude_and_clip_bp','Amplitude + clipping','#5544a5')]):
        axes[1].bar(xx+(k-1)*width,q[col],width,label=label,color=color)
    axes[1].set(xticks=xx,xticklabels=['16','32','64'],xlabel='States',ylabel='Mean absolute component (bp)',title='BIQAE error components, 97.5% risk')
    axes[1].legend(frameon=False)
    fig.savefig(BASE/'overview.png',dpi=180);plt.close(fig)
    report=f'''# 五项实验总报告：修正经典基线后的方案可行性

日期：2026-09-15。所有数值来自保存后重载的训练模型、策略及实际有限采样运行。

## 结论

**方案在小规模、理想模拟的混合训练与离散风险估计上可行；“BIQAE训练优于MC”不成立，当前也没有相对强经典基线的查询优势证据。主要精度瓶颈是分箱。**

| 用户问题 | 实验回答 |
|---|---|
| 修正后仍接近参考解吗？ | 是。30种子BIQAE测试ES相对精确梯度高0.05702%；与MC差值区间包含0。 |
| 同误差、同成功率谁省查询？ | 当前16状态任务QMC最省；5bp目标4096次，BIQAE约5.3万–5.6万次。 |
| 误差主要来自哪里？ | 分箱低估约18.9–25.4bp；BIQAE振幅/裁剪误差约0.5–0.9bp。 |
| 是否仅适用于一个市场窗口？ | 三个独立顺序测试窗口均接近参考，但没有证明未来市场保证或经典优势反转。 |
| 实现、网格或先验造成假象吗？ | 本次交叉验证未发现明显实现偏差；错配先验使误差增加，仍达已设目标。 |

![实际测量汇总]({(BASE/'overview.png').as_posix()})

左图是观测成功率筛选，MC为99/100，其余100/100；各自查询单位为经典向量样本或A/A†调用，不能解读为同等硬件耗时。右图的绝对误差项仅用于量级比较，不是可相加的总误差。

## 1. 修正后的30种子训练对照

{training_table}

各采样方法30种子，共150个策略及3个参考。每步8192预算、120步、相同约束和公共学习率0.3；经典方法复用完整收益向量。已存在的50个修正后策略逐文件哈希核验后复用，新增100个训练。全部保存策略重载评估。

BIQAE−MC平均差为+0.1524bp，配对95% bootstrap区间[-0.7885,+1.1499]bp；没有显著优越证据。BIQAE均值区间相对精确梯度的上界差0.42242%，低于运行前约定的5%接近门槛。该门槛表示宽松的工程可行性，不表示相等或优胜；bootstrap仅量化训练随机性。LP在离散训练目标上求解后再到测试集评估，未用测试集求最优权重。

训练共享精确VaR与分数尾部处理，因此是混合方法。精确阈值经典计算另记账，不包含在振幅查询量中。详情：{link('exp1_training/报告.md','实验1报告')}。

## 2. 同误差与可靠性

3000次全网格试验；下表为单配置95%成功率下界≥95%时的平均查询/预算：

{query_table}

100/100的双侧95%成功率下界为96.38%，99/100只有94.55%。对15个候选同时校正后，100个种子仍不足以认证整张网格的95%可靠性。区间覆盖99%–100%并不保证区间足够窄；QMC的5bp点误差与区间半宽联合要求在4096预算也通过，MC/IS保守区间在本预算网格内未通过。

该比较针对训练后的**离散分布真值**，没有涵盖分箱误差。QMC利用完整已知分布的排序/一维逆CDF结构；16状态精确枚举也已可行。详情及覆盖/宽度表：{link('exp2_precision/报告.md','实验2报告')}。

## 3. VaR与离散化消融

{ablation_table}

同一训练模型、生成样本及组合权重，900组配对诊断。增加状态数未消除约19–21bp的偏差；应先改进尾部压缩，再谈5/10bp总误差。真值VaR诊断针对离散分布，原始样本精确枚举另作分箱参考。详情：{link('exp3_ablation/报告.md','实验3报告')}。

## 4. 三个窗口重新训练与测试

{windows_table}

每窗口重新训练3个GMM候选，只按验证对数密度选模型；公共学习率只按验证集选，测试集不参与。5种算法各10种子，共150个新策略、9个GMM候选；测试窗口顺序划分。三个窗口BIQAE相对参考差的bootstrap上界均低于0.9%，满足预设5%接近门槛。历史窗口数量有限，窗口间并非独立市场抽样。详情：{link('exp4_windows/报告.md','实验4报告')}。

## 5. 已知振幅、官方IQAE与完整电路

完成1470次已知振幅、21次官方Qiskit IQAE、24次完整电路估计及24次配对解析对照，共1539次正式估计。BIQAE网格2049/8193，先验Beta(1,1)、(0.5,0.5)、错配(8,2)均完整运行。已测振幅误差均≤0.005；该单位是振幅，不能当作ES的5bp。

4283个轮次中象限分支不一致0次；8/16状态完整电路最大概率偏差3.14×10⁻¹²，24组成对点估计与查询量完全一致。代表电路最多5量子比特、深度2221、CX1497。官方IQAE停止规则与幂选择不同，仅作实现交叉验证，不能据其查询量宣称公平性能优势。

完整电路执行Statevector演化后有限shots采样，包含加载、受控旋转、反计算与Grover，使用查表载荷而非通用损失算术寄存器。未验证噪声硬件或实际端到端加速。第一次零载荷归一化失败及修正后全部重跑记录均保留。详情：{link('exp5_quantum/实验5报告.md','实验5报告')}。

## 完成规模、时间与复核

| 实验 | 最终完成规模 | 记录的计算墙钟 | 用户时间上限 |
|---|---|---:|---:|
| 1 | 150采样策略+3参考；其中50策略哈希复用 | 52.53秒（新增计算及评估） | 50分钟 |
| 2 | 3000次，30配置各100种子 | 20.49秒 | 60分钟 |
| 3 | 900组，30配置各30种子 | 11.45秒 | 35分钟 |
| 4 | 9个GMM候选+150采样策略+9参考 | 79.68秒 | 45分钟 |
| 5 | 1539次正式估计；首次失败另存 | 13.09秒（正式重跑） | 30分钟 |

这些时间是各脚本记录的计算时间，不含代码开发、审计和报告制作，也不是量子硬件时间。通过并行实验、复用已验证模型和解析有限shots模拟节约时间，无需跑满预算上限。

模型、训练轨迹、逐种子数据、区间、查询账本、协议和SHA256已保存。最终完整性结果见{link('final_audit.json','审计清单')}；算法测试执行结果另存test_results.txt。

### 复现入口

在项目目录、准备好的虚拟环境中运行：

```powershell
./.venv-biqae/Scripts/python.exe -m q3_biqae.suite_training --experiment all
./.venv-biqae/Scripts/python.exe -m q3_biqae.suite_risk precision
./.venv-biqae/Scripts/python.exe -m q3_biqae.suite_risk ablation
./.venv-biqae/Scripts/python.exe -m q3_biqae.suite_quantum_validation
./.venv-biqae/Scripts/python.exe -m q3_biqae.suite_report
```

实验输出已冻结，训练/实验脚本会拒绝覆盖；完整重跑应在副本中把输出常量改为新的目录，保留原数据与配置。报告生成器可在现有测量上重复执行。

**最终判断：保留方案作为理想小规模可行性原型；撤回训练优于MC的结论；现有证据不足以支持量子查询优势或5/10bp端到端风险精度。**
'''
    (BASE/'总报告.md').write_text(report,encoding='utf-8')
    for directory in [p, BASE/'exp3_ablation']:
        write_json(directory/'checksums.json',{str(f.relative_to(directory)):digest(f) for f in directory.rglob('*') if f.is_file() and f.name!='checksums.json'})
    write_json(BASE/'final_audit.json',{
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'existing_manifest_entries_verified':audited,'precision_rows':len(r2),'ablation_rows':len(r3),
        'precision_and_ablation_failed_trials':0,'all_checked_risk_budgets_within_cap':True,
        'max_decomposition_residual_bp':float(residual),
        'reporting_only_fix':'Bonferroni raw p value clipped at 1; original trials unchanged',
        'files':{str(f.relative_to(BASE)):digest(f) for f in BASE.rglob('*') if f.is_file() and f.name not in ['final_audit.json','test_results.txt']}})
    print(BASE/'总报告.md')

if __name__=='__main__':
    run()
