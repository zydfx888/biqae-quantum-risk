import json
import pandas as pd
import numpy as np
from .config import guarded, write_json

def save_optimization(directory, result):
    directory = guarded(directory)
    method = result["method"]
    rows = []
    for step in result.pop("logs"):
        for query_index, query in enumerate(step["queries"]):
            for round_index, row in enumerate(query["rounds"]):
                rows.append({"step": step["step"], "query": query_index, "round": round_index,
                             "component": query.get("component", str(query.get("asset"))),
                             "sign": query.get("sign", 0), "details_json": json.dumps(row, allow_nan=False)})
    # Posterior arrays belong in compressed tables, not multi-hundred-MB JSON.
    pd.DataFrame(rows).to_parquet(directory/f"optimization_{method}_rounds.parquet", compression="zstd", index=False)
    write_json(directory/f"optimization_{method}.json", result)

def save_estimates(directory, estimates, truth):
    directory = guarded(directory)
    rows, rounds, resources = [], [], []
    for estimate in estimates:
        key = {"method": estimate["method"], "seed": estimate["seed"]}
        rows.append({**key, **{k: v for k, v in estimate.items() if k not in {"resources", "logs", "ci"}},
                     "var_error": abs(estimate["var"]-truth["var"]), "es_error": abs(estimate["es"]-truth["es"]),
                     "var_ci_low": estimate["ci"]["var"][0], "var_ci_high": estimate["ci"]["var"][1],
                     "es_ci_low": estimate["ci"]["es"][0], "es_ci_high": estimate["ci"]["es"][1],
                     "var_covered": estimate["ci"]["var"][0] <= truth["var"] <= estimate["ci"]["var"][1],
                     "es_covered": estimate["ci"]["es"][0] <= truth["es"] <= estimate["ci"]["es"][1]})
        resources.append({**key, **estimate["resources"]})
        for query, log in enumerate(estimate["logs"]):
            for round_index, row in enumerate(log["rounds"]):
                rounds.append({**key, "query": query, "stage": log["stage"], "round": round_index,
                               "threshold": log["threshold"], "details_json": json.dumps(row, allow_nan=False)})
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory/"estimates.parquet", index=False)
    pd.DataFrame(resources).to_parquet(directory/"resources.parquet", index=False)
    pd.DataFrame(rounds).to_parquet(directory/"round_logs.parquet", index=False)
    frame.to_csv(directory/"estimates.csv", index=False)
    return frame

def plot_results(directory, frame, optimizations=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = guarded(directory)
    figures = directory/"figures"
    figures.mkdir(exist_ok=True)
    costs = pd.read_parquet(directory/"resources.parquet")
    data = frame.merge(costs[["method", "seed", "oracle_A_calls"]], on=["method", "seed"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, group in data.groupby("method"):
        cost = group.oracle_A_calls + group.samples
        axes[0].scatter(cost, group.es_error, label=name, s=16)
        axes[1].scatter(cost, group.var_error, s=16)
    axes[0].set(xlabel="A/A-dagger calls or classical samples", ylabel="Absolute ES error")
    axes[1].set(xlabel="A/A-dagger calls or classical samples", ylabel="Absolute VaR error")
    data.groupby("method")[["var_covered", "es_covered"]].mean().plot.bar(ax=axes[2])
    axes[2].set(ylabel="Empirical interval coverage", ylim=(0, 1.05))
    axes[0].legend(fontsize=7)
    fig.savefig(figures/"risk_resources.png", dpi=150)
    plt.close(fig)
    if optimizations:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for item in optimizations:
            axes[0].plot([h["cvar"] for h in item["history"]], label=item["method"])
        axes[0].set(xlabel="Iteration", ylabel="CVaR of averaged weights")
        axes[0].legend(fontsize=7)
        for item in optimizations:
            if item["method"] == "beta_biqae":
                axes[1].plot([h["average_weights"] for h in item["history"]])
                break
        axes[1].set(xlabel="Iteration", ylabel="Beta-BIQAE averaged weights")
        fig.savefig(figures/"optimization.png", dpi=150)
        plt.close(fig)

def report(directory, config, truth, frame, optimizations=None):
    directory = guarded(directory)
    lines = ["# BIQAE 风险估计运行报告", "", f"后端：`{config.backend}`；冻结分布：{config.dataset}；状态数：{config.states}；α：{config.alpha}",
             "", f"精确 VaR：{truth['var']:.8f}；精确 ES：{truth['es']:.8f}", "",
             "| 方法 | VaR 平均绝对误差 | ES 平均绝对误差 | VaR 覆盖率 | ES 覆盖率 | 收敛比例 |", "|---|---:|---:|---:|---:|---:|"]
    for name, group in frame.groupby("method"):
        lines.append(f"| {name} | {group.var_error.mean():.6g} | {group.es_error.mean():.6g} | {group.var_covered.mean():.1%} | {group.es_covered.mean():.1%} | {group.converged.mean():.1%} |")
    lines += ["", "![风险与资源](figures/risk_resources.png)", "", "## 解释边界", "",
              "- 本运行保留所有预算耗尽和未收敛结果；VaR 区间跨越 α 时不强行选方向。",
              "- MC/IS 使用有限样本 Hoeffding 区间；QMC 使用独立 scrambling 重复区间，并保留一维 Sobol 差异度误差界。",
              "- IQAE 使用逐轮 CP 区间和失败概率分配；MLAE 使用渐近 profile-likelihood 区间。",
              "- BIQAE 输出可信区间；Normal/Beta 为工程矩匹配近似，不能直接继承论文的覆盖率或复杂度定理。",
              "- analytic_shots 是经典计算理想测量概率的查询模型模拟；其电路资源字段为 0，表示未测量，不表示零成本。",
              "- circuits 中的电路资源为真实 Qiskit 转译结果，单独统计加载、编译和查表成本；无 QRAM 假设。",
              "- 少量种子的 smoke/MVP 结果不用于认定查询优势；需正式配对网格和误差匹配验收。",
              "- 未执行真实硬件；不据此声称端到端加速。"]
    if optimizations:
        lines += ["", "## CVaR 优化传导", "", "| 方法 | CVaR | LP 最优性差距 | 权重 L1 距离 | 约束违反 | 样本外 ES |", "|---|---:|---:|---:|---:|---:|"]
        for item in optimizations:
            lines.append(f"| {item['method']} | {item['cvar']:.6g} | {item['optimality_gap']:.6g} | {item['weight_l1_distance']:.6g} | {item['constraint_violation']:.3g} | {item.get('out_of_sample_es', 0):.6g} |")
        lines += ["", "![优化轨迹](figures/optimization.png)", "", "精确参考和采样梯度共享初始化、步长、投影与固定迭代次数；真值只用于评估，不用于采样梯度的阈值更新。"]
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
