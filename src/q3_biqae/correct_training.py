"""Retrain classical policies with shared samples, reusing frozen v1 inputs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import sys
import numpy as np
import pandas as pd
from .config import ROOT, Config, guarded, write_json, digest
from .scenario_table import ScenarioTable
from .feasibility import policy_train, evaluate, say


def run(source, destination):
    source, destination = guarded(source), guarded(destination)
    if destination.exists():
        raise FileExistsError("Use a new correction directory; do not overwrite frozen results")
    hashes = json.loads((source/"checksums.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        if digest(source/name) != expected:
            raise ValueError(f"Source artifact hash mismatch: {name}")
    destination.mkdir(parents=True)
    # Copy only frozen inputs and unaffected numerical results. Old reports,
    # classical policies and their summaries are not copied as new evidence.
    files = ["protocol.json", "evaluation_protocol.json", "data_manifest.json", "model_selection.json",
             "optimizer_selection.json", "frozen_distribution.npz", "generated_scenarios.npz",
             "train_returns.csv", "validation_returns.csv", "test_returns.csv", "risk_results.csv",
             "risk_rounds.parquet", "truths.json", "circuit_verification.json"]
    files += [f.name for f in source.glob("circuit_k*.qpy")]
    files += [str(f.relative_to(source)) for f in (source/"models").iterdir()
              if not f.name.startswith(("policy_mc_", "policy_sobol_", "policy_is_"))]
    reused = {}
    for name in files:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
        reused[name] = digest(target)
    p = json.loads((destination/"protocol.json").read_text())
    selection = json.loads((destination/"optimizer_selection.json").read_text())
    amendment = {"source_run": str(source.relative_to(ROOT)), "created_utc": datetime.now(timezone.utc).isoformat(),
                 "reason": "Prior classical policy training split budget into eight independently sampled signed components",
                 "correction": "MC/QMC/IS sample one batch of complete return vectors per optimization step, all gradient components reuse it",
                 "IS_proposal": "0.5*p + 0.5*normalize(p*(0.1+norm(return_vector*tail)/max_norm))",
                 "IS_normalization": "ordinary p/q weighting; no self-normalization",
                 "QMC": "one scrambled Sobol net of 8192 vectors per step, no eight-way division or replicate division",
                 "unchanged": ["GMM parameters", "scenario table", "split dates", "initial weights", "learning rate",
                               "step count", "per-step total budget", "seed list", "quantum policies", "independent VaR/ES results"],
                 "learning_rate": selection["learning_rate"], "test_retuning": False,
                 "retrained_methods": ["mc", "sobol", "is"], "reused_file_hashes": reused,
                 "timing_note": "Classical training timed in correction run; quantum timing reused from original run, not a synchronized hardware benchmark"}
    write_json(destination/"training_correction.json", amendment)
    write_json(destination/"environment.json", {"python": sys.version, "source_hashes": {f.name: digest(f) for f in (ROOT/"src/q3_biqae").glob("*.py")}})
    table = ScenarioTable.load(destination/"frozen_distribution.npz")
    cfg = Config(alpha=p["portfolio_alpha"], budget=p["portfolio_budget_per_step"], optimization_steps=p["portfolio_steps"],
                 epsilon=.005, grid_size=2049, max_rounds=16, shots_per_round=128,
                 max_grover_power=15, record_posterior=False)
    for method in ["mc", "sobol", "is"]:
        for seed in p["portfolio_seeds"]:
            say(f"Corrected shared-vector training {method}, seed={seed}, N={cfg.budget}/step")
            result = policy_train(table, cfg, method, seed, selection["learning_rate"], destination/"models"/f"policy_{method}_{seed}.json")
            if result["training_resources"]["samples"] != cfg.budget*cfg.optimization_steps:
                raise RuntimeError("Unexpected classical sample accounting")
    models = destination/"models"
    write_json(destination/"training_complete.json", {"status": "complete", "model_sha256": digest(models/"trained_risk_model.npz"),
               "table_sha256": table.sha256, "test_used_for_training_or_selection": False,
               "policy_hashes": {f.name: digest(f) for f in models.glob("policy_*.json")},
               "classical_gradient_sampling": "shared_return_vector_batch"})
    # Evaluation reloads the newly trained serialized classical policies.
    evaluate(destination, portfolio_only=True)
    current = pd.read_csv(destination/"portfolio_test_results.csv")
    previous = pd.read_csv(source/"portfolio_test_results.csv")
    comparison = current.merge(previous, on=["method", "seed"], suffixes=("_corrected", "_previous"))
    comparison.to_csv(destination/"training_before_after.csv", index=False)
    unchanged = {name: digest(destination/name)==expected for name, expected in reused.items()}
    if not all(unchanged.values()):
        raise RuntimeError("An unaffected frozen artifact was modified")
    write_json(destination/"reuse_verification.json", unchanged)
    correction_report(destination, comparison)
    write_json(destination/"checksums.json", {str(f.relative_to(destination)): digest(f) for f in destination.rglob("*") if f.is_file() and f.name != "checksums.json"})
    return destination


def correction_report(directory, comparison):
    from .feasibility import LABELS
    summary = comparison.groupby("method")[["test_es_previous", "test_es_corrected", "training_runtime_s_previous", "training_runtime_s_corrected"]].mean()
    lines = ["# 组合训练经典基线修正报告", "", "## 修正内容", "",
             "原训练代码把总预算分给4资产×正负载荷，并让经典方法独立采样。这削弱了经典基线；此前根据该实现作出的训练效果优劣比较应撤回，以本次修正后结果为准。", "",
             "MC/QMC/IS现在每步只采一批完整收益向量，同时计算全部有符号梯度。每步8192个向量、120步、10个种子；每个训练模型总计983,040次情景采样，不按资产数重复计费。IS采用一个公共向量载荷提议分布与p/q校正。", "",
             "冻结的GMM、分布、学习率0.3、初始化、数据切分和测试集全部沿用；重新训练30个经典组合模型，保存并重新加载后测试。IQAE/BIQAE训练模型、独立VaR/ES风险结果和电路验证均按哈希原样复用。未重新按测试结果调参。", "",
             "## 修正前后（测试ES越低越好）", "",
             "| 方法 | 原测试ES | 修正后测试ES | 原训练秒数 | 修正后训练秒数 |", "|---|---:|---:|---:|---:|"]
    for method in ["mc", "sobol", "is", "iqae", "grid_biqae", "exact", "lp"]:
        row = summary.loc[method]
        lines.append(f"| {LABELS[method]} | {row.test_es_previous*100:.5f}% | {row.test_es_corrected*100:.5f}% | {row.training_runtime_s_previous:.3f} | {row.training_runtime_s_corrected:.3f} |")
    quantum = summary.loc["grid_biqae", "test_es_corrected"]
    mc = summary.loc["mc", "test_es_corrected"]
    lines += ["", "## 更新后的结论", "",
              f"BIQAE 相对修正后 MC 的测试ES差距为 **{(quantum/mc-1)*100:+.3f}%**（正值表示BIQAE风险更高）。这是当前10个训练种子的点估计，不据此声称统计显著优越。",
              "方案在当前小规模理想模拟下能完成训练并接近精确梯度基线的可行性证据仍可保留；先前基于独立分量采样而得到的经典训练比较不能用于论证量子优势。", "",
              "经典训练计时来自本次修正运行，量子计时复用上次记录；表中计时用于说明实现成本，不是同时段受控硬件性能实验。LP计时0是原记录未计时。", "",
              "独立VaR/ES估计未改变：该部分此前已经复用样本。新增的共享批次修正也已接入通用joint-CVaR优化器，其阈值梯度与权重梯度共享同一批样本。", "",
              "## 复现", "", "```powershell",
              r".\.venv-biqae\Scripts\python.exe -m q3_biqae.correct_training --source artifacts/feasibility_trained_v1 --destination artifacts/feasibility_corrected_new",
              "```", "", "[更新后的完整报告](实验结果报告.md) · [逐种子前后对照](training_before_after.csv) · [冻结产物复用核验](reuse_verification.json)"]
    (directory/"基线修正报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="artifacts/feasibility_trained_v1")
    parser.add_argument("--destination", default="artifacts/feasibility_trained_v2")
    args = parser.parse_args()
    say(str(run(args.source, args.destination)))


if __name__ == "__main__":
    main()
