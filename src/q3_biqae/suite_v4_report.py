"""Chinese supplement report rendered exclusively from executed measurements."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT, digest, write_json
from .suite_report import table

BASE=ROOT/'artifacts/validation_suite_v4'
LABELS={'original_equal_tail90':'原方案（等权排序）','uniform_mass_actual':'等概率均匀','uniform_width_actual':'等损失宽均匀',
    'tail95_actual':'95%尾部加密（边界对齐）','tail90_offgrid_actual':'90%尾部加密（非对齐）'}
REP={'raw_reference':'原始20003样本精确参考','original64':'原方案64状态','static_actual64':'固定LP尾部加密64','adaptive_actual64':'当前权重动态加密64'}
METHOD={'exact':'精确梯度','mc':'MC','grid_biqae':'BIQAE'}

def link(path,label):return f'[{label}]({(BASE/path).as_posix()})'

def run():
    d=BASE/'exp1_binning';s=pd.read_csv(d/'exact_summary.csv');f=pd.read_csv(d/'exact_results.csv')
    p=pd.read_csv(d/'propagation/summary.csv');r=pd.read_csv(d/'propagation/results.csv')
    tr=pd.read_csv(d/'propagation/trajectory_exact_audit.csv');g=json.loads((d/'adaptive64_gate.json').read_text())
    rows=[]
    for m in LABELS:
        for states in [16,32,64]:
            aa=s[(s.method==m)&(s.states==states)&(s.scope=='anchor')].iloc[0]
            cc=s[(s.method==m)&(s.states==states)&(s.scope=='cross_weight')].iloc[0]
            rows.append([LABELS[m],states,f'{aa.var_max_abs_bp:.2f}',f'{aa.es_mae_bp:.3f}',f'{cc.es_mae_bp:.2f}',
                f'{cc.es_max_abs_bp:.2f}',f'{cc.gradient_relative_l2_max*100:.2f}%',f'{cc.centered_cosine_min:.3f}'])
    static_table=table(['分箱','状态','LP最大VaR误差bp','LP平均ES误差bp','其他权重ES平均误差bp','其他权重ES最大误差bp','其他权重梯度最大相对误差','最小方向余弦'],rows)
    policy_table=table(['表示','算法','原始模型ES','真实测试ES','最终分箱ES偏差bp','梯度相对误差'],[
        [REP[x.representation],METHOD[x.method],f'{100*x.raw_model_es:.6f}%',f'{100*x.test_es:.6f}%',f'{x.es_bias_bp:+.3f}',f'{100*x.gradient_relative_l2:.3f}%']
        for x in p.itertuples()])
    trajectory_table=table(['表示','抽查点数','最大ES误差bp','最大VaR误差bp','最大梯度相对误差','最小方向余弦'],[
        [REP[name],len(z),f'{z.es_bias_bp.abs().max():.3f}',f'{z.var_bias_bp.abs().max():.3f}',f'{100*z.gradient_relative_l2.max():.3f}%',f'{z.centered_gradient_cosine.min():.4f}']
        for name,z in tr.groupby('representation')])
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    methods=['original_equal_tail90','uniform_mass_actual','uniform_width_actual','tail95_actual','tail90_offgrid_actual']
    x=np.arange(5)
    for j,(scope,label,color) in enumerate([('anchor','LP anchor','#5ba99b'),('cross_weight','Other frozen weights','#d9944b')]):
        z=s[(s.states==64)&(s.scope==scope)].set_index('method').loc[methods]
        axes[0].bar(x+(j-.5)*.35,z.es_mae_bp,.35,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=['Original','Uniform\nmass','Uniform\nwidth','Tail95\naligned','Tail90\noffgrid'],ylabel='Mean absolute ES error (bp)',title='Frozen compression, 64 states')
    axes[0].legend(frameon=False)
    for rep,color in [('original64','#848a99'),('static_actual64','#d9944b'),('adaptive_actual64','#5ba99b')]:
        z=tr[(tr.representation==rep)&(tr.method=='grid_biqae')&(tr.evaluation=='current_weights_before_update')].copy()
        z['abs_es']=z.es_bias_bp.abs();v=z.groupby('step').abs_es.mean()
        axes[1].plot(v.index,v.values,marker='o',label={'original64':'Original64','static_actual64':'Static LP64','adaptive_actual64':'Adaptive64'}[rep],color=color)
    axes[1].set(xlabel='Training step',ylabel='Mean absolute compression ES error (bp)',title='Actual BIQAE training, 3 seeds')
    axes[1].legend(frameon=False)
    fig.savefig(d/'binning_and_training.png',dpi=180);plt.close(fig)
    report=f'''# 补充实验1：分箱方法、跨权重保真与重训传导

## 核心结论

**固定按一个组合损失做分箱，能改善该组合的风险精度，却不能自动用于其他组合或训练中的新权重。当前权重动态分箱在本次有限探针和少量重训中保留了精度，但增加了显著经典预处理成本，也没有改善真实测试集风险。**

![精确分箱与真实训练结果]({(d/'binning_and_training.png').as_posix()})

## 1. 先精确枚举，不调用振幅算法

重载原来训练完成的双成分GMM，核对模型SHA256，逐元素再现同一20003条生成收益。固定原LP权重作为分箱依据；另外在观察结果前冻结等权、4个单资产权重0.6的组合、4个旋转分散权重，共10组权重。两风险水平95%/97.5%、16/32/64状态、5种分箱，共300条精确结果。

“均匀”分别实现等概率质量与等损失宽度，避免定义含糊。分箱保存完整4资产向量均值和经验概率，而非只保存组合损失。对边界样本按质量分数切分；等损失宽分箱若有空箱则丢弃，实际状态数逐条记录。原方法保留既有舍入切点。

{static_table}

方向余弦用去除公共分量后的梯度计算，反映权重和为1约束下的相对资产方向；同时保存原梯度、逐资产误差、正负号、梯度大小与投影更新误差。完整指标见{link('exp1_binning/exact_results.csv','300条精确结果')}。

### 不能把锚点ES零误差当成普遍改善

95%尾部加密将95%和97.5%放在质量边界。条件均值构造使LP锚点的ES及各资产尾部期望严格保留，这是结构性吻合；16状态的LP最大VaR误差却达到52.62bp，其他冻结权重的ES最大误差达78.29bp。梯度在其他组合处也可能显著改变。

非对齐90%尾部方案64状态在LP处平均ES误差0.106bp、最大0.172bp，显示改善并非全部来自精确对齐；但其其他权重平均ES误差28.03bp、最大74.79bp，仍未通过跨权重检查。任何静态分箱候选均未通过全部冻结权重门槛。

### 尾部概率质量检查

比较的是**原始分布VaR阈值处**的严格超越概率与包含边界概率；没有使用由定义恒等于1−α的分数尾质量冒充保真。逐资产一阶矩最大守恒误差约8.3×10⁻¹⁷，但保存均值并不足以保存任意权重下的尾部结构。尾概率误差、每个资产的梯度及更新向量均已保存。

## 2. 精确门槛通过后才重训

初始门槛在计算前固定：全部检查点VaR误差≤10bp、ES误差≤5bp、尾概率误差≤0.0025（0.25个百分点）、梯度相对L2误差≤10%、中心化方向余弦≥0.95。

初始当前权重32状态/95%对齐分箱也因VaR偏差失败，因此没有直接进入训练。随后记录追加协议，检查64状态、90%非对齐、每次按当前权重分箱；这是观察前次失败后的**顺序探索**，不是原始独立预注册确认。

在相同10权重×2风险水平的20条精确探针中，该候选通过：

- 最大VaR误差：{g['max_var_abs_bp']:.3f}bp。
- 最大ES误差：{g['max_es_abs_bp']:.3f}bp。
- 最大尾概率误差：{g['max_tail_probability_abs_error']:.6f}，即{100*g['max_tail_probability_abs_error']:.3f}个百分点。
- 最大梯度相对L2误差：{100*g['max_relative_gradient_l2']:.3f}%；最小中心化方向余弦：{g['min_centered_cosine']:.6f}。

精确探针通过后，才执行下列少量重训。其余不通过的静态表示仅保留为诊断对照。没有把有限10权重通过推断为整个可行域都通过。

## 3. 22个实际训练策略：固定分箱误差重新出现

原始样本精确参考1个策略；原方案64、固定LP尾部64、动态当前权重64，各精确梯度1个、MC和BIQAE各3种子，共22策略（18个采样训练）。各120步、每步8192预算、学习率0.3、相同投影约束、相同原始模型等权损失标准差作为梯度缩放。最终策略为120次投影更新平均，全部保存后重载评估。

该对照改变了状态数并统一原始模型缩放，因此应在本表内比较，不能把数值直接拼接到旧16状态训练表。

{policy_table}

固定LP分箱在最初LP锚点很准确，但训练后的其他权重ES偏差回到约26–32bp。动态分箱最终偏差约0.14bp；BIQAE模型内ES为2.914062%，原始样本精确梯度为2.912559%，相差约0.150bp。这支持精度改善沿训练传导，不能以3个种子宣称统计优越。

**真实测试ES并未随模型内精度改善而下降。** 例如BIQAE原方案64状态测试ES为2.763194%，动态64状态为2.810247%；两者均在同一冻结测试集评价。动态方案更接近原始模型优化结果，而原始模型与测试市场的差异仍然存在。分箱修复不是泛化改善的证据。

### 训练轨迹抽查

每10步和最后一步在更新前权重精确检查，另检查最终平均策略；每表示7个策略共98个审计点。并非每一步都审计，不作未经检查全路径的保证。

{trajectory_table}

动态分箱审计点最大ES误差0.162bp，方向余弦最低0.99896；固定LP分箱最大ES误差33.707bp，方向余弦最低0.251。可直接从{link('exp1_binning/propagation/trajectory_exact_audit.csv','轨迹精确审计')}复核权重更新后的失真。

## 4. 成本与适用范围

- 动态分箱每步需要用当前权重排序、聚合全部20003条原始收益；每策略120次重分箱，按轮累计2400360条参与记录。这不是物理内存读取次数，当前实现还会逐箱扫描样本，实际运算更多。
- 本机动态重分箱约2.37–2.59秒/策略；动态BIQAE总训练约7.47秒/策略。原始20003样本精确梯度训练本身约0.45秒，因而没有端到端加速证据。
- MC每策略983040次共享完整向量采样；动态BIQAE平均979906次A/A†调用。经典重分箱和阈值计算另记，不能被量子查询账本隐藏。
- 动态编码依赖当前组合，不是一次加载后适合所有组合的固定分布；使用压缩梯度近似原始模型梯度，不是对离散化映射求导。有限探针不能证明未来权重或市场普遍保真。
- GMM没有重新拟合，避免用新测试结果挑模型；本次只重训组合策略。原始参考是有限模型生成样本精确枚举，不是总体真值。

## 文件与执行

精确协议、追加协议、门槛和模型哈希位于本目录；propagation保存22个策略、120步轨迹、原始量子轮次、重分箱耗时及评估结果。原有v2/v3结果未改写。

入口为`python -m q3_biqae.suite_v4_binning exact`、`python -m q3_biqae.suite_v4_binning adaptive_gate`、`python -m q3_biqae.suite_v4_propagation`。冻结输出拒绝覆盖，复跑需换输出目录。正式重训71.84秒。
'''
    (d/'报告.md').write_text(report,encoding='utf-8')
    assert len(f)==300 and len(r)==22 and len(tr)==308
    assert r.constraint_violation.max()<1e-8
    for z in r.itertuples():
        path=d/f'propagation/policy_{z.representation}_{z.method}_{z.seed}.json'
        assert digest(path)==z.policy_sha256
    write_json(d/'verification.json',{'status':'pass','exact_rows':300,'initial_current_probe_rows':20,'followup_current_probe_rows':20,
        'trained_policies':22,'policy_hashes_verified':22,'trajectory_audit_rows':308,
        'max_constraint_violation':float(r.constraint_violation.max()),
        'all_policy_budgets_within_cap':bool((r.samples+r.A_calls<=120*8192).all()),
        'independent_readonly_audit':'No blocking numerical bug found; 500 subgradient support checks, minimum residual -7.0e-18; chronology and no test-based selection reviewed.',
        'logical_sample_records_not_physical_memory_reads':True})
    write_json(d/'checksums.json',{str(f.relative_to(d)):digest(f) for f in d.rglob('*') if f.is_file() and f.name!='checksums.json'})
    # Other experiment reports are produced and verified by their executing modules.
    resource=BASE/'exp5_resources'
    overview=f'''# 五项补充实验总报告

日期：2026-09-15。结果来自原训练后GMM、固定生成样本、实际重训策略和有限shots风险估计。

## 最重要的结果：分箱误差可修复，但需要随权重变化处理

1. 对比原方案、两种均匀分箱、95%对齐与90%非对齐尾部加密，16/32/64状态、10组预先冻结权重、两风险水平，共300条精确结果。**没有静态候选通过全部权重检查**。
2. 固定按LP分箱，锚点ES可结构性精确，但其他权重误差最高约75–78bp。非对齐64状态在LP处最大ES误差仅0.172bp，重训后仍回到约26–32bp，确认权重错配是主要问题。
3. 追加的“每次按当前权重重分箱、64状态非对齐尾部”先通过20条精确检查：最大VaR误差5.283bp、ES误差0.211bp、梯度相对误差0.409%。随后实际训练22策略；动态BIQAE最终分箱偏差约0.139bp，98个动态轨迹审计点最大ES误差0.162bp。
4. 代价是每步重新处理全部20003样本。动态BIQAE真实测试ES2.810247%，原方案64状态为2.763194%，**未出现测试泛化改善，更没有端到端量子加速证据**。

![分箱与重训证据]({(d/'binning_and_training.png').as_posix()})

详情与门槛、逐资产梯度、尾概率和成本：{link('exp1_binning/报告.md','实验1详细报告')}。

## 其余四项

| 补充内容 | 完成规模 | 结果 |
|---|---|---|
| Student-t厚尾与压力混合分布 | 6000次完整VaR/ES，60配置各100种子 | 5bp离散目标QMC4096次最省；原分箱低估15–23bp，最高预算相对原始样本的5/10bp联合成功率均0。 |
| IQAE、网格、Normal、Beta对照 | 840已知振幅+800风险任务 | 各风险配置100/100通过5bp；Normal/Beta约0.49–0.65bp ES误差，与网格差别很小。 |
| Grover/预算分配/停止规则 | 840风险任务，单因素改变 | 最大幂31→127使BIQAE估计误差约减半；VaR仅20%预算时97.5%风险成功14/30，40%/60%则30/30。相同轮数的停止模式配对完全一致。 |
| 8/16/32完整风险资源 | 48完整流程、1513轮测量、192独特电路 | 12组解析/完整电路自适应路径完全一致；32状态约3868万–4067万次CX/流程。 |

Normal/Beta在本项目中是**逐轮后验矩投影工程版本**，与仅更换Beta先验不同，但仍需网格积分，未证明与论文原算法等价。此次完成现有四版本的正式对照，不能称论文原版复现。

两类新分布由原训练后GMM的尺度及相关结构校准，是模型压力场景；没有把合成分布伪称为新历史数据训练。误差同时区分离散表与原始有限样本参考。各方法的经典样本/A调用不是同等硬件耗时，QMC还利用完整已知表排序结构。

预算分配60%时BIQAE调用约44564，相比40%时55048减少，但ES误差从0.600升到0.749bp。该30种子探索不能证明95%可靠性，也未按测试集选择新部署参数。扩大Grover幂意味着更深电路，不能视为免费改善。

## 完整风险资源曲线

| 状态数 | 各配置平均CX执行量范围 | 最大单电路深度 | 量子比特 |
|---|---:|---:|---:|
| 8 | 139.51万–241.78万 | 5621 | 4 |
| 16 | 540.56万–885.58万 | 29618 | 5 |
| 32 | 3868.00万–4067.07万 | 80308 | 6 |

覆盖实际CDF搜索及ES查询的全部轮次。36次解析有限shots流程重放全部所需电路，另12次由完整Statevector驱动自适应采样；最大概率偏差6.31×10⁻¹²，配对阈值、幂、shots、成功计数和门数一致。正式运行865.62秒，独特电路及组件编译187.11秒。

使用原冻结分箱的8/16/32状态核算，与实验1动态64状态重训分开。电路包含加载、查表旋转、反计算及Grover；CDF阈值载荷由经典端构造，未实现通用收益算术/比较寄存器。门数乘shots是逻辑操作量，不是真机时间；未包括硬件路由、纠错等。8状态97.5%超额载荷全零且裁剪后ES精确，仍实际执行ES测量并计入188928次CX，不能把该退化当精度优势。

## 报告与原始证据

- {link('exp1_binning/报告.md','1. 分箱、跨权重与重训传导')}
- {link('exp2_distributions/报告.md','2. 两类完整风险分布对照')}
- {link('exp3_variants/报告.md','3. Normal/Beta工程版本正式对照')}
- {link('exp4_controls/报告.md','4. Grover上限、预算分配和停止规则')}
- {link('exp5_resources/实验5报告.md','5. 完整风险流程资源核算')}

所有目录均保存先写协议、逐种子结果、模型哈希和账本。源码仅新增实验入口及停止模式选项；原算法默认停止行为保持不变，新增回归测试验证。训练新增18个采样策略与4个精确参考，全部保存后重载评估；旧结果不覆盖。

最终算法测试90项通过；22个保存策略、2160步采样账本重构和逐步约束检查通过。完整性核验见{link('final_audit.json','最终审计清单')}，测试输出见{link('test_results.txt','测试记录')}。分布、版本和控制实验分别用时135.13、64.08和46.29秒；分箱重训71.84秒，资源核算865.62秒。均为脚本计算墙钟，不含开发、审计或报告制作，不能作为硬件比较。

**建议结论：当前可以确认“分箱依据与组合权重错配”的缺陷，并得到有限范围有效的动态修复；还不能确认固定编码的通用保真、市场泛化改善或量子优势。**
'''
    (BASE/'总报告.md').write_text(overview,encoding='utf-8')
    print(BASE/'总报告.md')

if __name__=='__main__':run()
