"""Build one report, plan removal of superseded documents, verify retained evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
OUT=ROOT/'综合实验报告.md'
META=ART/'report_consolidation'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def key(p):return str(p.resolve()).casefold()
def link(p,title):return f'[{title}]({p.resolve().as_posix()})'

CHAPTERS=[
 ('修正经典基线后的30种子训练','validation_suite_v3/exp1_training/报告.md'),
 ('三个时间窗口重新训练与测试','validation_suite_v3/exp4_windows/报告.md'),
 ('同误差查询量、覆盖率与可靠性','validation_suite_v3/exp2_precision/报告.md'),
 ('厚尾Student-t与压力混合分布','validation_suite_v4/exp2_distributions/报告.md'),
 ('原分箱下的VaR误差与状态数消融','validation_suite_v3/exp3_ablation/报告.md'),
 ('分箱方法、跨权重保真与重训传导','validation_suite_v4/exp1_binning/报告.md'),
 ('已知振幅、官方IQAE与完整电路交叉验证','validation_suite_v3/exp5_quantum/实验5报告.md'),
 ('Normal/Beta工程版本正式对照','validation_suite_v4/exp3_variants/报告.md'),
 ('Grover上限、预算分配和停止规则','validation_suite_v4/exp4_controls/报告.md'),
 ('8/16/32状态完整风险流程资源','validation_suite_v4/exp5_resources/实验5报告.md'),
]

def build():
    META.mkdir(exist_ok=True)
    sources=[(title,ART/path) for title,path in CHAPTERS]
    sources.extend([('历史：10种子经典基线修正',ART/'feasibility_trained_v2/基线修正报告.md'),
        ('历史：首轮训练后模型完整实验',ART/'feasibility_trained_v2/实验结果报告.md'),
        ('历史：工程初始验证记录',ROOT/'VALIDATION.md')])
    sources.extend((f'历史冒烟运行：{p.parent.name}',p) for p in sorted((ART/'runs').glob('*/report.md')))
    remove=sorted([p for p in ART.rglob('*') if p.is_file() and p.suffix.lower() in {'.md','.html','.pdf','.docx'}]+[ROOT/'VALIDATION.md'])
    anchors={key(p):f'section-{i}' for i,(_,p) in enumerate(sources,1)}
    for p in remove:anchors.setdefault(key(p),'overview')
    evidence=[]
    def transform(text,source):
        def target(m):
            prefix,raw=m.groups();raw=raw.strip().strip('<>')
            if raw.startswith(('http://','https://','mailto:','data:','#')):return m.group(0)
            # All source report links are plain filesystem references, without title attributes.
            base,_,fragment=raw.partition('#')
            path=Path(base) if Path(base).is_absolute() else source.parent/base
            if key(path) in anchors:return prefix+'#'+anchors[key(path)]+')'
            return prefix+path.resolve().as_posix()+('#'+fragment if fragment else '')+')'
        text=re.sub(r'(!?\[[^\]\n]*\]\()([^\)\n]+)\)',target,text)
        lines=text.splitlines()
        if lines and lines[0].startswith('# '):lines=lines[1:]
        # Preserve fenced code verbatim; normalize remaining headings beneath chapter headings.
        fenced=False;out=[]
        for line in lines:
            if line.startswith('```'):fenced=not fenced
            if not fenced and re.match(r'^#{1,5} ',line):line='#'+line
            out.append(line)
        return '\n'.join(out).strip()
    header=f'''# 方案三：全部实验结果综合报告

汇总日期：2026-09-15。本文是统一结果入口，替代此前分散的总报告、简报和单项报告。数值整理自已保存的实际运行结果；本次汇总没有重新训练或重新采样，也没有生成新的实验数值。

<a id="overview"></a>

## 核心判断

**方案在小规模理想模拟下能够完成风险估计和混合组合训练；现有证据不支持优于强经典基线，也不支持真机量子优势。最主要缺陷是分箱依据与组合权重错配。动态分箱在有限检查中修复了该误差，但没有改善真实测试集表现，并增加了经典计算成本。**

| 核心问题 | 当前有效结果 | 判断 |
|---|---|---|
| 修正经典采样后，BIQAE还能接近参考吗？ | 30种子测试ES：BIQAE 2.727519%，MC 2.725995%，精确梯度2.725964%，LP 2.687108% | 接近参考；BIQAE−MC配对差+0.1524bp，95%区间[-0.7885,+1.1499]bp，无优越证据 |
| 同误差下谁省查询？ | 原16状态两风险水平5bp目标：QMC 4096次；BIQAE约53056–55572次 | 当前已知查表任务QMC更省；抽象样本/A调用不是同等硬件时间 |
| 原19–25bp误差来自哪里？ | 原分箱ES低估18.9–25.4bp，振幅误差通常不到1bp | 输入压缩是主要瓶颈 |
| 按实际组合分箱就足够吗？ | 固定LP尾部64状态锚点ES最大误差0.172bp；其他权重最大误差74.79bp；重训后约26–32bp | 固定编码不保留其他组合的尾部风险与梯度 |
| 动态分箱是否有效？ | 64状态随当前权重重分箱，精确探针最大ES误差0.211bp；少量重训最终误差约0.14bp | 有限范围的模型内保真改善，非通用编码保证 |
| 模型内改善是否传到真实测试集？ | 新64状态对照中，BIQAE测试ES由原方案2.763194%变为动态方案2.810247% | 未改善；不可把分箱保真等同于市场泛化 |
| 完整电路成本如何？ | 32状态每次完整风险流程各配置平均约3868万–4067万CX，最大深度80308 | 远不止查询计数；没有硬件加速证据 |

## 口径、数据与执行真实性

- **确实训练的对象**是收益分布模型和组合权重策略。IQAE/BIQAE是估计算法；后验更新不等于训练神经网络。
- 主数据为已下载的French五行业日收益表，取Cnsmr、Manuf、HiTec、Hlth四资产；原方案六行业下载不可用，替代来源与原始文件SHA256已记录。训练2000–2017年4528行，验证2018–2020年756行，测试2021–2025年1255行。双成分全协方差GMM从3个EM初始化按验证对数密度选择。
- 主模型保存后重载，按固定种子生成20003条完整收益向量。风险估计通常固定LP16权重[0.4,0,0,0.6]。多时间窗口分别重新训练模型；新厚尾/压力分布是由训练模型尺度校准的合成压力场景，不是新增真实市场训练集。
- **训练结果**来自实际权重更新、保存参数与重新加载评价。30种子扩展对照中，50个已修正策略经哈希验证后复用，新增100个；三个窗口新增150个采样策略；分箱修复新增18个采样策略及4个精确参考。不同轮次有重用与配对，不能相加当成独立样本数。
- **批量量子算法实验**使用解析振幅概率驱动随机有限shots采样，实际执行自适应幂选择、后验更新与停止流程。完整电路部分实际运行Qiskit Statevector；每轮从电路概率抽取有限shots，没有逐shot演化状态向量。
- **没有真机实验。** 正式训练、可靠性和完整资源研究是理想模拟；早期曾运行小型Aer去极化噪声冒烟测试，见历史附录，不能当作正式噪声鲁棒性验收。
- 查询单位：经典方法为一次完整收益向量采样，复用所有资产梯度或VaR/ES；量子为Σshots×(2k+1)，包括A与A†。动态分箱、完整表排序和精确阈值计算的经典成本另列。
- 1bp=0.0001损失率=0.01个百分点。必须区分**离散表真值、原始20003条生成样本精确参考、真实历史测试集**；5bp离散误差达标并不表示端到端总误差达标。
- 100/100成功的单配置双侧95% Clopper–Pearson下界约96.38%，99/100约94.55%，30/30约88.43%；全网格同时认证更严格。贝叶斯可信区间不能直接当作频率置信保证。训练种子bootstrap也不是未来市场置信保证。
- Normal/Beta是逐轮矩投影的**工程近似**，并非仅更换初始Beta先验；仍需网格积分，尚未证明等价于论文原版或其理论复杂度。

## 实验清单与证据强度

| 主题 | 实际规模 | 结果位置 |
|---|---|---|
| 修正训练基线 | 5方法各30种子、150策略+3参考；50策略哈希复用 | 第1节 |
| 多窗口泛化 | 3窗口、9个GMM候选、150采样策略+9参考 | 第2节 |
| 原训练模型等误差/可靠性 | 3000次，30配置各100种子 | 第3节 |
| 两类新风险分布 | 6000次，60配置各100种子 | 第4节 |
| 真值VaR与状态数消融 | 900组、16/32/64状态，同一模型及权重 | 第5节 |
| 分箱方法与传导 | 300条静态精确结果+40条当前权重探针、22个训练策略、308个轨迹审计点 | 第6节 |
| 量子实现交叉验证 | 1539次：1470已知振幅、21官方IQAE、24完整电路+24解析对照 | 第7节 |
| Normal/Beta四版本 | 840已知振幅+800完整风险任务 | 第8节 |
| Grover/分配/停止消融 | 840完整风险任务，单因素改变 | 第9节 |
| 完整风险资源 | 48流程、1513轮、25种载荷、192独特电路 | 第10节 |

所有正式运行中的异常与早期失败均保留；未以删除不利种子提高成功率。部分正式重跑为0失败，不表示此前没有程序错误。修复包括经典VaR置信上界、MLAE预算边界和零载荷实验脚本处理，具体见正文与历史附录。

## 导航

'''
    parts=[header]
    for i,(title,p) in enumerate(sources,1):parts.append(f'- [{i}. {title}](#section-{i})\n')
    for i,(title,p) in enumerate(sources,1):
        raw=p.read_text(encoding='utf-8-sig');evidence.append({'chapter':i,'source':str(p.relative_to(ROOT)),'source_sha256':sha(p)})
        notice=''
        if i>10:
            notice='> **历史结果，仅用于追溯。** 本节保留当时的配置、样本数和观察值；“本次”“未完成”等表述均指该历史阶段。训练主结论以第1节30种子结果为准，风险可靠性以第3–4节为准，最新测试为90项通过。初期错误经典梯度采样不能用于证明优势。\n\n'
        parts.append(f'\n\n<a id="section-{i}"></a>\n\n## {i}. {title}\n\n{notice}原始证据目录：{link(p.parent,p.parent.name)}。本节正文已并入本文，原单项报告可删除。\n\n'+transform(raw,p))
    parts.append(f'''\n\n## 统一结论与后续边界

目前完成的是计算与实现可行性验证：修正后BIQAE接近参考，量子概率/电路一致性通过，动态分箱在有限探针及重训中降低模型内误差。尚未证明：相对强经典方法优势、真实测试泛化改善、静态编码对任意组合保真、论文Normal/Beta逐式等价、带真实硬件成本的加速。

代码最近完整测试为90项通过；22个补充策略、2160步采样账本及逐步约束已独立重构。此前83/63项测试记录属于当时版本，不应当作最新总数。

### 文档整理与复核

原方案文档、README、IMPLEMENTATION、源码、配置、环境、原始市场数据、训练模型、生成样本、逐次CSV/Parquet、QPY电路、失败记录及图片均保留。仅删除被本文取代的旧报告文档与旧验证摘要。

旧checksums和final_audit属于**运行时历史快照**，其中可能包含此次有意删除的旧报告路径；不改写其历史哈希。删除前后的文件与哈希记录见{link(META/'cleanup_manifest.json','文档清理清单')}，当前保留证据的完整性检查见{link(META/'verification.json','汇总后核验')}。旧报告生成器再次执行可能重新生成单项报告；若要重跑实验，应使用新的输出目录以免覆盖冻结证据。

本次只是重新汇总与整理文档，没有把旧实验重新计为新运行，也没有修改任何模型参数、原始观测值或失败记录。
''')
    OUT.write_text(''.join(parts),encoding='utf-8')
    write(META/'sources.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'report':str(OUT),'chapters':evidence,'report_sha256':sha(OUT)})
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'workspace':str(ROOT.resolve()),'report':str(OUT.resolve()),
          'reason':'Superseded report documents consolidated into one report; preserve experimental evidence.',
          'files':[{'path':str(p.resolve()),'relative':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,
                    'represented_by':anchors[key(p)]} for p in remove]}
    write(META/'cleanup_manifest.json',plan)
    print(json.dumps({'report':str(OUT),'chapters':len(sources),'delete_documents':len(remove),'bytes':OUT.stat().st_size},ensure_ascii=False))

def verify():
    manifest=json.loads((META/'cleanup_manifest.json').read_text(encoding='utf-8'))
    deleted={key(Path(x['path'])):x for x in manifest['files']};ok=0;removed_refs=0;bad=[]
    for suite in ['validation_suite_v3','validation_suite_v4']:
        base=ART/suite;snapshot=json.loads((base/'final_audit.json').read_text(encoding='utf-8'))
        for name,expected in snapshot['files'].items():
            p=base/name
            if p.is_file():
                if sha(p)!=expected:bad.append({'path':str(p),'reason':'hash mismatch'})
                else:ok+=1
            elif key(p) in deleted and deleted[key(p)]['sha256']==expected:removed_refs+=1
            else:bad.append({'path':str(p),'reason':'unexpected missing file'})
    text=OUT.read_text(encoding='utf-8');broken=[]
    for target in re.findall(r'!?\[[^\]\n]*\]\(([^\)\n]+)\)',text):
        if target.startswith(('http://','https://','mailto:','#')):continue
        p=Path(target.split('#',1)[0].strip('<>'))
        if not p.is_absolute():p=ROOT/p
        if not p.exists():broken.append(str(p))
    # verification.json is this function's output, so its first-run forward link is expected.
    broken=[p for p in broken if key(Path(p))!=key(META/'verification.json')]
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'report_sha256':sha(OUT),
        'retained_v3_v4_snapshot_entries_verified':ok,'historical_report_entries_accounted_for':removed_refs,
        'unexpected_evidence_changes':bad,'broken_report_file_links':sorted(set(broken)),
        'deleted_reports':sum(not Path(x['path']).exists() for x in manifest['files']),
        'planned_reports':len(manifest['files'])}
    result['status']='pass' if not bad and not broken else 'failed'
    write(META/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='report_sha256'},ensure_ascii=False))
    if bad or broken:raise RuntimeError('Consolidation verification failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','verify']);a=parser.parse_args()
    build() if a.mode=='build' else verify()
