"""Generate plain-language conclusions directly from recorded experiment rows."""
import json
from pathlib import Path
import html
import numpy as np
import pandas as pd
from scipy.stats import beta
from .config import guarded, write_json
from .exact_reference import risk
from .feasibility import LABELS, METHODS

def cp(successes, total):
    return [0. if successes == 0 else float(beta.ppf(.025, successes, total-successes+1)),
            1. if successes == total else float(beta.ppf(.975, successes+1, total-successes))]

def block_bootstrap_ratio(test_returns, weights, reference, repetitions=500, block=20):
    rng = np.random.default_rng(2026091601)
    n = len(test_returns)
    ratios = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, int(np.ceil(n/block)))
        index = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        x = test_returns[index]
        w = weights[int(rng.integers(len(weights)))]
        p = np.full(n, 1/n)
        numerator = risk(-x@w, p, .975)["es"]
        denominator = risk(-x@reference, p, .975)["es"]
        ratios.append(numerator/denominator-1)
    return np.quantile(ratios, [.025, .975]).tolist()

def make_report(directory):
    directory = guarded(directory)
    p = json.loads((directory/"protocol.json").read_text())
    manifest = json.loads((directory/"data_manifest.json").read_text())
    selection = json.loads((directory/"model_selection.json").read_text())
    optimizer = json.loads((directory/"optimizer_selection.json").read_text())
    circuits = json.loads((directory/"circuit_verification.json").read_text())
    truths = json.loads((directory/"truths.json").read_text())
    risk_rows = pd.read_csv(directory/"risk_results.csv")
    portfolio = pd.read_csv(directory/"portfolio_test_results.csv")
    summaries = []
    for (alpha, budget, method), rows in risk_rows.groupby(["alpha", "budget", "method"]):
        summaries.append({"alpha": alpha, "budget": int(budget), "method": method,
                          "var_mae_bp": rows.var_abs_error.mean()*10000, "es_mae_bp": rows.es_abs_error.mean()*10000,
                          "es_mae_sd_bp": rows.es_abs_error.std()*10000,
                          "joint_success_rate": rows.joint_success.mean(), "success_rate_cp95": cp(int(rows.joint_success.sum()), len(rows)),
                          "var_coverage": rows.var_covered.mean(), "es_coverage": rows.es_covered.mean(),
                          "mean_queries": rows.cost.mean(), "mean_shots": rows.shots.mean(),
                          "mean_runtime_ms": rows.runtime_ms.mean(),
                          "var_interval_width_bp": rows.var_interval_width.mean()*10000,
                          "es_interval_width_bp": rows.es_interval_width.mean()*10000,
                          "var_resolved_rate": rows.var_resolved.mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(directory/"risk_summary.csv", index=False)
    policy_summary = portfolio.groupby("method").agg(test_es=("test_es", "mean"), test_es_sd=("test_es", "std"),
                     test_var=("test_var", "mean"), train_s=("training_runtime_s", "mean"),
                     training_A_calls=("training_A_calls", "mean"), training_samples=("training_samples", "mean"),
                     constraint=("constraint_violation", "max"), predicted_es=("predicted_es", "mean"),
                     exceedance=("var_exceedance_rate", "mean")).fillna(0)
    equal_es = policy_summary.loc["equal_weight", "test_es"]
    exact_es = policy_summary.loc["exact", "test_es"]
    lp_es = policy_summary.loc["lp", "test_es"]
    policy_summary["improvement_vs_equal_pct"] = (1-policy_summary.test_es/equal_es)*100
    policy_summary["gap_vs_exact_pct"] = (policy_summary.test_es/exact_es-1)*100
    policy_summary.to_csv(directory/"portfolio_summary.csv")
    test_returns = pd.read_csv(directory/"test_returns.csv", index_col=0).values
    exact_weight = np.array(json.loads(portfolio.loc[portfolio.method=="exact", "weights"].iloc[0]))
    quantum_weights = np.array([json.loads(w) for w in portfolio.loc[portfolio.method=="grid_biqae", "weights"]])
    ratio_ci = block_bootstrap_ratio(test_returns, quantum_weights, exact_weight)
    risk_gate_by_alpha = {str(alpha): bool(np.any((summary.method=="grid_biqae") & (summary.alpha==alpha) & (summary.joint_success_rate>=.95))) for alpha in p["risk_alphas"]}
    risk_gate = all(risk_gate_by_alpha.values())
    quantum_es = policy_summary.loc["grid_biqae", "test_es"]
    policy_gate = bool(quantum_es<=1.05*exact_es and policy_summary.loc["grid_biqae", "constraint"]<=1e-8)
    circuit_gate = bool(max(circuits["oracle"]["probability_error"], circuits["oracle"]["inverse_error"],
                           max(r["probability_error"] for r in circuits["amplified"])) <= 1e-8)
    gates = {"risk_observed_success_gate": risk_gate_by_alpha, "portfolio_point_gate": policy_gate,
             "portfolio_block_bootstrap_relative_gap_ci95": ratio_ci, "portfolio_ci_upper_within_5pct": ratio_ci[1]<=.05,
             "oracle_circuit_gate": circuit_gate, "limited_feasibility_supported": risk_gate and policy_gate and circuit_gate,
             "formal_quantum_advantage_established": False}
    write_json(directory/"conclusions.json", gates)
    feasible = gates["limited_feasibility_supported"]
    conclusion = "在本次小规模理想模拟条件下，支持方案的计算可行性。" if feasible else "本次实验尚未满足预先设定的全部可行性条件，不能宣布方案已验证可行。"
    max_budget = max(p["risk_budgets"])
    primary = summary[(summary.budget==max_budget) & (summary.alpha==.975)].set_index("method")
    best_es_method = primary.es_mae_bp.idxmin()
    best_policy_method = policy_summary.loc[["mc", "sobol", "is", "iqae", "grid_biqae"]].test_es.idxmin()
    corrected = (directory/"training_correction.json").exists()
    correction_note = (
        "**组合训练基线修正版：** MC/QMC/IS 每步共享一批8192个完整收益向量计算全部梯度，"
        "经典训练已重新运行30个模型；冻结GMM、学习率、测试数据、量子训练结果和独立VaR/ES估计原样复用。"
        "旧版独立分量采样的训练比较已撤回。"
        f"BIQAE相对修正后MC的测试ES差距为 {(quantum_es/policy_summary.loc['mc','test_es']-1)*100:+.3f}%"
        "（正值表示BIQAE风险更高）；不据小差异认定统计优势。"
    )
    lines = ["# 方案三最小可行性实验：训练后模型与基线对照", "", "## 一、直接结论", "",
             f"**{conclusion}**", "",
             f"- 训练后的 BIQAE 组合在独立测试期的 ES 为 **{quantum_es*100:.4f}%**，相对未优化等权组合降低 **{(1-quantum_es/equal_es)*100:.2f}%**。",
             f"- 相对同协议精确梯度训练基线，测试 ES 差距为 **{(quantum_es/exact_es-1)*100:+.2f}%**；20 日区块 Bootstrap 的 95% 相对差距区间为 **[{ratio_ci[0]*100:+.2f}%, {ratio_ci[1]*100:+.2f}%]**。",
             f"- 在 α=97.5%、查询上限 {max_budget:,} 的风险估计中，ES 平均误差最小的是 **{LABELS[best_es_method]}**。BIQAE 的结果见下表，不能据训练成功推断其优于经典基线。",
             "- **本实验不认定量子加速或全面优于基线。** 批量对照使用理想测量概率模拟；另以真实 Qiskit 完整电路验证计算链。", "",
             "## 二、这次确实训练了什么", "",
             "1. 用真实日收益训练二分量高斯混合风险模型（EM 迭代学习混合概率、均值和协方差）。训练标准化参数仅来自训练期。",
             "2. 三个初始化均完成训练，只按验证期对数似然选择模型；模型保存为 `models/trained_risk_model.npz` 后重新加载，生成并冻结 16 个收益状态。",
             "3. 在同一已训练模型输出的分布上，分别训练 MC、QMC、IS、IQAE、BIQAE 的组合权重；每种采样方法 10 个训练种子，每次 120 次权重更新。",
             "4. 测试程序重新读取保存的权重文件，在完全未参与训练与选择的真实测试期计算 VaR/ES。没有把精确真值填作算法预测。", "",
             "**IQAE/BIQAE 是风险估计器，本身没有神经网络权重。这里的训练对象是收益分布模型和组合权重策略。** BIQAE 使用网格后验版，避免用未校准的 Normal/Beta 近似代替。", "",
             "## 三、数据与最小实验设置", "",
             "数据：[Kenneth French 官方行业组合数据](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)。固定选择五行业表中的 Cnsmr、Manuf、HiTec、Hlth 四个行业，日收益百分数统一除以 100。原方案所写六行业地址返回 404，调整已记录来源和 SHA256。", "",
             "| 数据分段 | 日期 | 样本数 | 用途 |", "|---|---|---:|---|"]
    for name, label, usage in [("train", "训练", "学习分布模型参数"), ("validation", "验证", "选择初始化和公共学习率"), ("test", "测试", "只输出最终结果")]:
        split = manifest["splits"][name]
        lines.append(f"| {label} | {split['first']}—{split['last']} | {split['rows']} | {usage} |")
    lines += ["", f"固定设置：16 状态，α=95%/97.5%，风险估计每组 30 个配对种子，查询上限 {p['risk_budgets']}；优化使用 α=97.5%，权重和为 1、单资产权重不超过 60%。公共学习率 {optimizer['learning_rate']} 仅由验证集选择。", "",
              "量子查询按 `shots×(2k+1)` 计算，包含 A 和 A†。经典预算为分布样本数。统一的是查询上限，实际停止调用数另列；不能把 shots 当作全部调用数。", "",
              "优化采用最小混合结构：经典端计算当前 VaR 和边界原子质量，量子端估计正负尾部期望，再由经典投影更新权重。该实验验证量子梯度组件能否接入训练，不代表全量子优化。", "",
              "## 四、训练后模型的风险估计对照", "",
              "这里比较的是**同一已训练模型输出的冻结分布**，参考值为其精确枚举。真实测试期的预测/实际差异另列，不混入量子计算误差。1 bp=0.01 个百分点；预先固定达标标准为 VaR 和 ES 绝对误差均≤20 bp。", "",
              f"主表固定查询上限 {max_budget:,}，报告全部方法，不按测试效果挑选配置。", "",
              "| α | 方法 | VaR误差(bp) | ES误差(bp) | 联合达标率 | ES区间覆盖率 | 实际平均查询 | 平均耗时(ms) |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for alpha in p["risk_alphas"]:
        for method in METHODS:
            row = summary[(summary.alpha==alpha)&(summary.budget==max_budget)&(summary.method==method)].iloc[0]
            lines.append(f"| {alpha:.1%} | {LABELS[method]} | {row.var_mae_bp:.3f} | {row.es_mae_bp:.3f} | {row.joint_success_rate:.1%} | {row.es_coverage:.1%} | {row.mean_queries:,.0f} | {row.mean_runtime_ms:.2f} |")
    lines += ["", "**达标率是本次观察值，不是成功概率保证。** 即使 30/30 达标，Clopper–Pearson 95% 区间下界也只有约 88.4%；5% 失败概率的严格验收仍需更多种子。IQAE 为 CP 区间，BIQAE 为可信区间；宽区间的覆盖率高不意味着估计效率高。全部区间宽度及预算结果见 `risk_summary.csv`。", "",
              "![风险误差和预算](figures/risk_comparison.png)", "",
              "### 不同预算下是否稳定", "", "| α | 查询上限 | BIQAE ES误差(bp) | BIQAE联合达标 | MC联合达标 | QMC联合达标 |", "|---:|---:|---:|---:|---:|---:|"]
    for alpha in p["risk_alphas"]:
        for budget in p["risk_budgets"]:
            group = summary[(summary.alpha==alpha)&(summary.budget==budget)].set_index("method")
            lines.append(f"| {alpha:.1%} | {budget:,} | {group.loc['grid_biqae','es_mae_bp']:.3f} | {group.loc['grid_biqae','joint_success_rate']:.1%} | {group.loc['mc','joint_success_rate']:.1%} | {group.loc['sobol','joint_success_rate']:.1%} |")
    lines += ["", "## 五、训练后的组合在真实测试期表现", "",
              "ES 是最差 2.5% 尾部对应的日损失均值，越低越好。表中每种采样方法为 10 次训练结果均值；精确 LP/精确梯度是确定性参考，不人为重复计成 10 份独立样本。", "",
              "| 方法 | 测试VaR(%) | 测试ES(%) | 比等权降低(%) | 与精确梯度ES差距(%) | 平均训练秒数 |", "|---|---:|---:|---:|---:|---:|"]
    for method in ["equal_weight", "lp", "exact", *METHODS]:
        row = policy_summary.loc[method]
        lines.append(f"| {LABELS[method]} | {row.test_var*100:.4f} | {row.test_es*100:.4f} | {row.improvement_vs_equal_pct:.2f} | {row.gap_vs_exact_pct:+.2f} | {row.train_s:.2f} |")
    lines += ["", "LP和等权行的训练耗时字段0代表该字段未计时，不表示求解没有成本。其他训练耗时包含经典阈值计算及量子测量模型模拟；不包含公共 GMM 训练和硬件执行。", "",
              "![训练与测试](figures/training_test.png)", "",
              "### 学到的权重", "", "| 方法 | Cnsmr | Manuf | HiTec | Hlth |", "|---|---:|---:|---:|---:|"]
    for method in ["lp", "exact", *METHODS]:
        selected_weights = np.array([json.loads(w) for w in portfolio.loc[portfolio.method==method, "weights"]]).mean(axis=0)
        lines.append(f"| {LABELS[method]} | "+" | ".join(f"{w:.2%}" for w in selected_weights)+" |")
    lines += ["", "## 六、模型误差与量子误差分开看", "",
              "固定采用已训练 LP 权重，下表展示：拟合模型生成的大样本、16状态离散模型、真实测试期三者的 ES。这是分布模型与离散化的误差，不应归因于 BIQAE。", "",
              "| α | 模型大样本ES(%) | 离散模型ES(%) | 真实测试ES(%) |", "|---:|---:|---:|---:|"]
    for truth in truths:
        lines.append(f"| {truth['alpha']:.1%} | {truth['generated_continuous']['es']*100:.4f} | {truth['frozen']['es']*100:.4f} | {truth['heldout_real']['es']*100:.4f} |")
    lines += ["", "静态模型跨多年测试存在分布漂移，历史数据库也可能修订。这不是逐日滚动预测或交易回测；不据此作收益承诺。", "",
              "## 七、实际电路验证与可行性判定", "",
              "| 检查项 | 本次结果 |", "|---|---|"]
    lines += [f"| 训练后模型 Oracle 概率误差 | {circuits['oracle']['probability_error']:.3e} |",
              f"| 逆电路误差 | {circuits['oracle']['inverse_error']:.3e} |",
              f"| Grover k=0/1/3/7 最大概率误差 | {max(r['probability_error'] for r in circuits['amplified']):.3e} |"]
    actual = circuits["actual_full_circuit_BIQAE"]
    lines += [f"| 完整电路 BIQAE 实测振幅 / 精确振幅 | {actual['estimate']:.8f} / {actual['truth']:.8f} |",
              f"| 完整电路振幅区间 | [{actual['interval'][0]:.8f}, {actual['interval'][1]:.8f}] |",
              f"| 完整电路最大深度 / 量子比特 | {actual['resources']['max_depth']} / {actual['resources']['qubits']} |", "",
              f"预先设定的风险精度条件：{'通过观察值检查' if risk_gate else '未全部通过'}；训练组合测试 ES 与精确梯度相差不超过 5% 且约束满足：{'通过' if policy_gate else '未通过'}；电路门限：{'通过' if circuit_gate else '未通过'}。", "",
              "**结论范围：** 有限种子、单一真实数据切分、4资产16状态的理想模拟可行性，不等于真机可行性、统计上已证明的量子优势或生产可用性。若强经典基线更准或更快，保留该结果。", "",
              "## 八、复现与交付物", "",
              "```powershell", r".\.venv-biqae\Scripts\python.exe -m q3_biqae.feasibility all --directory artifacts/feasibility_new", "```", "",
              "- `protocol.json`：执行前冻结的训练/验证/测试划分、指标和预算。",
              "- `models/trained_risk_model.npz`、`models/policy_*.json`：实际训练得到、并重新加载用于评价的模型及权重。",
              "- `model_selection.json`、`optimizer_selection.json`、训练轨迹和原始轮次：训练及选择证据。",
              "- `risk_results.csv` / `risk_summary.csv`、`portfolio_test_results.csv` / `portfolio_summary.csv`：全部结果。",
              "- `circuit_verification.json`、`circuit_k*.qpy`：真实 Qiskit 电路和转译资源。",
              "- `checksums.json`、`training_complete.json`：原始数据、模型与产物追溯。"]
    report = "\n".join(lines)+"\n"
    if corrected:
        lines[2:2] = [correction_note, ""]
        report = "\n".join(lines)+"\n"
    (directory/"实验结果报告.md").write_text(report, encoding="utf-8")
    brief = ["# 方案三实验结果简报", "", "## 结论", "",
             f"**{conclusion}但目前没有证明它整体优于经典基线。**", "",
             "本次使用真实行业日收益：2000–2017年训练，2018–2020年验证，2021–2025年独立测试。实际训练了风险分布模型和组合权重，保存后重新加载计算结果。风险估计做了900组对照；各采样方法做了10次组合训练，每次120步。", "",
             "## 1. 训练后的模型表现", "",
             "测试 ES 表示最差2.5%尾部的日均损失，越低越好。", "",
             "| 方法 | 独立测试ES |", "|---|---:|"]
    for method in ["equal_weight", "lp", "exact", *METHODS]:
        brief.append(f"| {LABELS[method]} | {policy_summary.loc[method,'test_es']*100:.4f}% |")
    brief += ["", f"BIQAE 比等权组合的测试风险降低 **{(1-quantum_es/equal_es)*100:.2f}%**；与精确梯度训练仅差 **{(quantum_es/exact_es-1)*100:+.2f}%**，但仍不如精确 LP。", "",
              "## 2. 相比基线，风险算得准不准", "",
              "统一查询上限65,536，置信水平97.5%，30个种子均值。1 bp=0.01个百分点，误差越小越好。", "",
              "| 方法 | ES估计误差(bp) | 实际平均查询 | 计算耗时(ms) |", "|---|---:|---:|---:|"]
    for method in METHODS:
        row = primary.loc[method]
        brief.append(f"| {LABELS[method]} | {row.es_mae_bp:.3f} | {row.mean_queries:,.0f} | {row.mean_runtime_ms:.2f} |")
    brief += ["", "BIQAE 在这个预算下比 MC 更准，但**QMC 更准且更快**；不能把量子方法称为最佳方案。", "",
              "## 3. 为什么支持可行性", "",
              "- 确实完成了训练、模型保存、重新加载和独立测试，不是仅跑通代码。",
              "- 最大预算下，BIQAE 在两个风险置信水平的各30次试验中，VaR和ES误差均不超过预定的20 bp。",
              f"- 训练后组合接近精确梯度基线，测试ES相对差距的区块Bootstrap 95%区间为 [{ratio_ci[0]*100:+.2f}%, {ratio_ci[1]*100:+.2f}%]，权重约束均满足。",
              "- 由训练后模型构建的5量子比特电路通过验证；实际完整电路BIQAE结果区间包含目标振幅。", "",
              "**限制：** α=97.5%时，BIQAE在4,096和16,384预算下联合达标率仅3.3%，需要更高预算才稳定；30个种子不能证明95%的真实成功概率。批量结果来自理想测量模拟，另有完整电路验证；经典端仍负责训练时的VaR阈值，未做真机实验。", "",
              "**建议：可作为小规模研究原型继续验证；当前优先保留QMC/精确LP为实际基线，不应据此承诺量子加速。**", "",
              "[完整实验报告](实验结果报告.md) · [风险结果原始数据](risk_results.csv) · [训练后组合测试数据](portfolio_test_results.csv)"]
    if corrected:
        brief[2:2] = [correction_note, ""]
    (directory/"结果简报.md").write_text("\n".join(brief)+"\n", encoding="utf-8")
    figures(directory, summary, portfolio, policy_summary)
    # Standalone readable HTML report, no server or external scripts required.
    body = []
    table_lines = []
    code = False
    for line in lines:
        if line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if all(set(c)<=set("-:") for c in cells):
                continue
            table_lines.append("<tr>"+"".join(f"<td>{c}</td>" for c in cells)+"</tr>")
            continue
        if table_lines:
            body.append("<div class='table'><table>"+"".join(table_lines)+"</table></div>")
            table_lines=[]
        if line.startswith("```"):
            code = not code
            continue
        if line.startswith("!["):
            source = line.split("](", 1)[1].rstrip(")")
            body.append(f"<img src='{html.escape(source)}'>")
        elif line.startswith("#"):
            level = len(line)-len(line.lstrip("#"))
            body.append(f"<h{level}>"+html.escape(line.lstrip("# "))+f"</h{level}>")
        elif line:
            body.append("<p>"+html.escape(line).replace("**", "")+"</p>")
    (directory/"实验结果报告.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>训练后可行性实验报告</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.8 'Microsoft YaHei',sans-serif;color:#183047}h1,h2{color:#064c64}h2{border-bottom:2px solid #e1edf0;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #d8e3e8;padding:8px}tr:first-child{background:#e8f3f5;font-weight:bold}.table{overflow:auto}img{width:100%;height:auto}p{overflow-wrap:anywhere}</style><body>"+"\n".join(body)+"</body></html>", encoding="utf-8")

def figures(directory, summary, portfolio, policy_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams["font.family"] = "Microsoft YaHei" if any(f.name=="Microsoft YaHei" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    output = directory/"figures"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for alpha, axis in zip([.95, .975], axes):
        for method in METHODS:
            rows = summary[(summary.alpha==alpha)&(summary.method==method)].sort_values("budget")
            axis.plot(rows.budget, rows.es_mae_bp, "o-", label=LABELS[method])
        axis.axhline(20, color="gray", linestyle="--", label="20 bp threshold")
        axis.set(xscale="log", yscale="log", xlabel="查询预算上限", ylabel="ES 平均绝对误差（bp）", title=f"置信水平 {alpha:.1%}")
    axes[0].legend(fontsize=8)
    fig.savefig(output/"risk_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for path in sorted((directory/"models").glob("gmm_*_training.json")):
        rows = json.loads(path.read_text())
        axes[0].plot([r["training_average_log_likelihood_standardized"] for r in rows], label=path.stem.split("_")[1])
    axes[0].set(title="风险模型真实训练轨迹", xlabel="EM 迭代", ylabel="训练平均对数似然")
    axes[0].legend(title="初始化种子", fontsize=8)
    for method in ["exact", "mc", "grid_biqae"]:
        path = next((directory/"models").glob(f"policy_{method}_*.json"))
        rows = json.loads(path.read_text())["trajectory"]
        axes[1].plot([r["train_es"]*100 for r in rows], label=LABELS[method])
    axes[1].set(title="组合权重训练轨迹（固定首种子）", xlabel="训练迭代", ylabel="训练 CVaR（%）")
    axes[1].legend(fontsize=8)
    names = ["equal_weight", "lp", "exact", *METHODS]
    axes[2].bar(range(len(names)), [policy_summary.loc[n, "test_es"]*100 for n in names])
    axes[2].set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=55, ha="right", fontsize=8)
    axes[2].set(title="独立真实测试期", ylabel="测试 ES（%，越低越好）")
    fig.savefig(output/"training_test.png", dpi=160)
    plt.close(fig)
