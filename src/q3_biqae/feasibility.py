"""Minimal train/freeze/test study. No test-set tuning or fabricated outcomes."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import argparse
import json
import sys
import uuid
import numpy as np
import pandas as pd
from .config import ROOT, Config, guarded, write_json, digest
from .data_prepare import parse_french
from .trained_model import train_mixture, TrainedRiskModel, freeze_model_scenarios
from .exact_reference import risk, cvar, gradient, tail_weights
from .classical_cvar import solve_cvar
from .optimizer import project_weights
from .amplitude import estimate_payload
from .es_estimation import estimate_risk
from .resource_ledger import ResourceLedger
from .scenario_table import ScenarioTable

METHODS = ["mc", "sobol", "is", "iqae", "grid_biqae"]
LABELS = {"equal_weight": "未优化等权", "lp": "精确LP基线", "exact": "精确梯度训练", "mc": "MC", "sobol": "Sobol QMC", "is": "重要性采样", "iqae": "IQAE", "grid_biqae": "BIQAE（网格）"}

def say(message):
    print(message, flush=True)

def protocol():
    return {
        "purpose": "Train a real-data risk model and portfolio policies; test feasibility rather than assume superiority",
        "train": ["2000-01-01", "2017-12-31"], "validation": ["2018-01-01", "2020-12-31"],
        "test": ["2021-01-01", "2025-12-31"], "assets": ["Cnsmr", "Manuf", "HiTec", "Hlth"],
        "risk_model": "two-component full-covariance Gaussian mixture, EM, training-only standardization",
        "model_initializations": [13, 29, 43], "selection": "highest validation log density; never test",
        "scenario_count": 20003, "frozen_states": 16, "scenario_seed": 20260915,
        "risk_alphas": [.95, .975], "risk_budgets": [4096, 16384, 65536],
        "risk_seeds": list(range(100, 130)), "portfolio_seeds": list(range(200, 210)),
        "portfolio_alpha": .975, "portfolio_steps": 120, "portfolio_budget_per_step": 8192,
        "learning_rates": [.03, .1, .3], "learning_rate_selection": "exact-gradient portfolio with lowest validation ES",
        "methods": METHODS, "risk_weights": "trained LP policy reloaded from checkpoint, common to every estimator",
        "risk_success": "absolute VaR and ES error <=0.002 (20 basis points), vs exact FROZEN trained-model truth",
        "risk_gate": "BIQAE observed joint success >=95% at some preregistered budget at each alpha; report CP uncertainty",
        "portfolio_gate": "BIQAE mean heldout ES <=1.05 * exact-gradient heldout ES; constraints <=1e-8",
        "circuit_gate": "trained-model Oracle target probability and inverse errors <=1e-8; amplified probabilities also verified",
        "limitations": "single real-data split, finite seeds, retrospective revised public data, ideal simulator, no hardware speed claims",
        "hybrid_training": "classical exact VaR and fractional atom selection; estimated signed tail expectations; all methods share this threshold rule",
        "classical_training_sampling": "one shared batch of complete return vectors per step for all assets; IS uses a common vector-norm proposal",
        "created_utc": datetime.now(timezone.utc).isoformat()
    }

def empirical_table(values):
    return ScenarioTable(values, np.full(len(values), 1/len(values)))

def policy_train(table, cfg, method, seed, learning_rate, path=None):
    started = perf_counter()
    w = np.full(4, .25)
    scale = max(float(np.std(table.losses(w))), 1e-6)
    average = np.zeros(4)
    ledger = ResourceLedger()
    trajectory, raw_rounds = [], []
    rng = np.random.default_rng(seed)
    for step in range(cfg.optimization_steps):
        if method == "exact":
            g = gradient(table, w, cfg.alpha)
        elif method in {"mc", "sobol", "is"}:
            from .classical_gradient import shared_batch_gradient
            tail = tail_weights(table, w, cfg.alpha)
            g, _, batch_ledger, batch_logs = shared_batch_gradient(
                table, tail, cfg.alpha, cfg.budget, int(rng.integers(2**31)), method)
            ledger.merge(batch_ledger)
            for query in batch_logs:
                for row in query["rounds"]:
                    raw_rounds.append({"step": step, "asset": -1, "sign": 0,
                                       "details": json.dumps(row, allow_nan=False)})
        else:
            tail = tail_weights(table, w, cfg.alpha)
            g = np.zeros(4)
            for asset in range(4):
                for sign in (1, -1):
                    value = np.maximum(sign*table.returns[:, asset], 0)
                    bound = float(value.max())
                    if bound == 0:
                        continue
                    result = estimate_payload(table.probabilities, value*tail/bound, method, cfg, cfg.budget//8,
                                              int(rng.integers(2**31)), epsilon=.005)
                    g[asset] -= sign*bound*result.estimate/(1-cfg.alpha)
                    ledger.merge(result.ledger)
                    for row in result.rounds:
                        raw_rounds.append({"step": step, "asset": asset, "sign": sign,
                                           "details": json.dumps(row, allow_nan=False)})
        eta = learning_rate/np.sqrt(step+1)
        w = project_weights(w-eta*g/scale, cfg.w_max)
        average += w
        wa = average/(step+1)
        trajectory.append({"step": step, "weights": wa.tolist(), "train_es": cvar(table, wa, cfg.alpha),
                           "A_calls": ledger.oracle_A_calls, "samples": ledger.samples})
    final = average/cfg.optimization_steps
    result = {"method": method, "seed": seed, "weights": final.tolist(), "learning_rate": learning_rate,
              "training_steps": cfg.optimization_steps, "training_runtime_s": perf_counter()-started,
              "training_resources": ledger.to_dict(), "training_es": cvar(table, final, cfg.alpha),
              "gradient_sampling": "shared_vector_batch" if method in {"mc", "sobol", "is"} else "exact" if method == "exact" else "signed_amplitude_queries",
              "classical_threshold_state_reads": cfg.optimization_steps*len(table.probabilities),
              "trajectory": trajectory}
    if path:
        path = guarded(path)
        write_json(path, result)
        pd.DataFrame(raw_rounds).to_parquet(path.with_suffix(".rounds.parquet"), compression="zstd", index=False)
    return result

def train(directory):
    directory = guarded(directory)
    directory.mkdir(parents=True, exist_ok=False)
    p = protocol()
    write_json(directory/"protocol.json", p)  # Written before fitting or evaluating.
    write_json(directory/"environment.json", {"python": sys.version, "source_hashes": {f.name: digest(f) for f in (ROOT/"src/q3_biqae").glob("*.py")}})
    source = ROOT/"data/raw/5_Industry_Portfolios_daily_CSV.zip"
    frame = parse_french(source)[p["assets"]]
    parts = {name: frame.loc[start:end] for name, (start, end) in ((n, p[n]) for n in ("train", "validation", "test"))}
    if any(len(f) < 250 for f in parts.values()):
        raise ValueError("Insufficient rows in chronological split")
    if not parts["train"].index.max() < parts["validation"].index.min() < parts["test"].index.min():
        raise ValueError("Invalid chronology")
    write_json(directory/"data_manifest.json", {"source": "Kenneth French 5 Industry Portfolios daily, value weighted",
               "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/5_Industry_Portfolios_daily_CSV.zip",
               "raw_sha256": digest(source), "assets": p["assets"], "splits": {n: {"rows": len(f), "first": str(f.index.min().date()), "last": str(f.index.max().date())} for n, f in parts.items()},
               "note": "Original scheme's six-industry download was 404; official five-industry first four selected before training."})
    for name, values in parts.items():
        values.to_csv(directory/f"{name}_returns.csv", index_label="date")
    models = directory/"models"
    models.mkdir()
    candidates = []
    for seed in p["model_initializations"]:
        say(f"EM training initialization {seed}")
        fitted, history = train_mixture(parts["train"].values, seed=seed)
        fitted.save(models/f"gmm_{seed}.npz")
        write_json(models/f"gmm_{seed}_training.json", history)
        candidates.append({"seed": seed, "iterations": len(history), "train_log_density": float(fitted.log_density(parts["train"].values).mean()),
                           "validation_log_density": float(fitted.log_density(parts["validation"].values).mean())})
    selected = max(candidates, key=lambda row: row["validation_log_density"])
    # Deliberately reload serialized learned parameters for all downstream results.
    fitted = TrainedRiskModel.load(models/f"gmm_{selected['seed']}.npz")
    fitted.save(models/"trained_risk_model.npz")
    write_json(directory/"model_selection.json", {"candidates": candidates, "selected": selected})
    scenarios = fitted.sample(p["scenario_count"], p["scenario_seed"])
    np.savez_compressed(directory/"generated_scenarios.npz", returns=scenarios)
    table = freeze_model_scenarios(scenarios, p["frozen_states"])
    table.save(directory/"frozen_distribution.npz")
    cfg = Config(alpha=p["portfolio_alpha"], budget=p["portfolio_budget_per_step"], optimization_steps=p["portfolio_steps"],
                 epsilon=.005, grid_size=2049, max_rounds=16, shots_per_round=128,
                 max_grover_power=15, record_posterior=False)
    validation = empirical_table(parts["validation"].values)
    lr_results = []
    for rate in p["learning_rates"]:
        policy = policy_train(table, cfg, "exact", 13, rate)
        lr_results.append({"learning_rate": rate, "validation_es": cvar(validation, policy["weights"], cfg.alpha)})
    rate = min(lr_results, key=lambda row: row["validation_es"])["learning_rate"]
    write_json(directory/"optimizer_selection.json", {"validation_only": lr_results, "learning_rate": rate})
    lp = solve_cvar(table, cfg.alpha, cfg.w_max)
    write_json(models/"policy_lp.json", {"method": "lp", "seed": 0, **lp})
    write_json(models/"policy_equal_weight.json", {"method": "equal_weight", "seed": 0, "weights": [.25]*4})
    for method in ["exact", *METHODS]:
        seeds = [p["portfolio_seeds"][0]] if method == "exact" else p["portfolio_seeds"]
        for seed in seeds:
            say(f"Portfolio training {method}, seed={seed}, steps={cfg.optimization_steps}")
            policy_train(table, cfg, method, seed, rate, models/f"policy_{method}_{seed}.json")
    write_json(directory/"training_complete.json", {"status": "complete", "model_sha256": digest(models/"trained_risk_model.npz"),
               "table_sha256": table.sha256, "test_used_for_training_or_selection": False,
               "policy_hashes": {f.name: digest(f) for f in models.glob("policy_*.json")}})
    return directory

def evaluate(directory, portfolio_only=False):
    directory = guarded(directory)
    # Frozen before opening heldout returns or producing benchmark outcomes.
    if not portfolio_only:
        write_json(directory/"evaluation_protocol.json", {
            "classical_risk_baselines": "one shared sampled distribution for VaR+ES, not repeated threshold sampling",
            "MC_interval": "DKW simultaneous CDF bound", "QMC_interval": "one-dimensional scrambled net discrepancy",
            "IS_interval": "self-normalized IS simultaneous Hoeffding ratio bounds",
            "IS_proposal": "0.5*p + 0.5*normalize(p*(0.05+normalized_frozen_loss**4))",
            "reason": "Use strong standard classical implementations; prevent repeated-query overhead from weakening baselines",
            "frozen_utc": datetime.now(timezone.utc).isoformat()})
    completed = json.loads((directory/"training_complete.json").read_text())
    p = json.loads((directory/"protocol.json").read_text())
    models = directory/"models"
    assert digest(models/"trained_risk_model.npz") == completed["model_sha256"]
    model = TrainedRiskModel.load(models/"trained_risk_model.npz")
    table = ScenarioTable.load(directory/"frozen_distribution.npz")
    assert table.sha256 == completed["table_sha256"]
    test_frame = pd.read_csv(directory/"test_returns.csv", index_col=0)
    test = empirical_table(test_frame.values)
    model_info = {"test_average_log_density": float(model.log_density(test.returns).mean())}
    weights = json.loads((models/"policy_lp.json").read_text())["weights"]
    policies = []
    for path in models.glob("policy_*.json"):
        assert digest(path) == completed["policy_hashes"][path.name]
        policy = json.loads(path.read_text())
        w = np.asarray(policy["weights"])
        r = risk(test.losses(w), test.probabilities, p["portfolio_alpha"])
        predicted = risk(table.losses(w), table.probabilities, p["portfolio_alpha"])
        policies.append({"method": policy["method"], "seed": policy["seed"], "weights": json.dumps(w.tolist()),
                         "test_var": r["var"], "test_es": r["es"], "predicted_es": predicted["es"],
                         "var_exceedance_rate": float(np.mean(test.losses(w)>predicted["var"])),
                         "constraint_violation": float(abs(w.sum()-1)+np.maximum(-w, 0).sum()+np.maximum(w-.6, 0).sum()),
                         "training_runtime_s": policy.get("training_runtime_s", 0),
                         "training_A_calls": policy.get("training_resources", {}).get("oracle_A_calls", 0),
                         "training_samples": policy.get("training_resources", {}).get("samples", 0)})
    pd.DataFrame(policies).to_csv(directory/"portfolio_test_results.csv", index=False)
    write_json(directory/"model_test.json", model_info)
    if portfolio_only:
        # Corrected training can be evaluated without rerunning or altering
        # the already-correct independent VaR/ES benchmark and circuit audit.
        from .feasibility_report import make_report
        make_report(directory)
        write_json(directory/"checksums.json", {str(f.relative_to(directory)): digest(f) for f in directory.rglob("*") if f.is_file() and f.name != "checksums.json"})
        return directory
    cfg = Config(alpha=.975, epsilon=.0002, grid_size=4097, max_rounds=32,
                 shots_per_round=256, max_grover_power=31, record_posterior=False)
    rows, logs = [], []
    npz = np.load(directory/"generated_scenarios.npz")
    continuous = empirical_table(npz["returns"])
    truths = []
    for alpha in p["risk_alphas"]:
        truth = risk(table.losses(weights), table.probabilities, alpha)
        full_truth = risk(continuous.losses(weights), continuous.probabilities, alpha)
        test_truth = risk(test.losses(weights), test.probabilities, alpha)
        truths.append({"alpha": alpha, "frozen": truth, "generated_continuous": full_truth, "heldout_real": test_truth})
        for budget in p["risk_budgets"]:
            for method in METHODS:
                say(f"Heldout trained-model risk benchmark alpha={alpha}, budget={budget}, {method}, 30 paired seeds")
                for seed in p["risk_seeds"]:
                    if method in {"mc", "sobol", "is"}:
                        from .feasibility_baselines import estimate_classical_risk
                        result = estimate_classical_risk(table, weights, alpha, budget, seed, method)
                    else:
                        result = estimate_risk(table, alpha, budget, seed, method, replace(cfg, alpha=alpha), weights)
                    rows.append({"alpha": alpha, "budget": budget, "method": method, "seed": seed,
                                 "var": result["var"], "es": result["es"],
                                 "var_abs_error": abs(result["var"]-truth["var"]), "es_abs_error": abs(result["es"]-truth["es"]),
                                 "joint_success": abs(result["var"]-truth["var"])<=.002 and abs(result["es"]-truth["es"])<=.002,
                                 "var_covered": result["ci"]["var"][0]<=truth["var"]<=result["ci"]["var"][1],
                                 "es_covered": result["ci"]["es"][0]<=truth["es"]<=result["ci"]["es"][1],
                                 "var_interval_width": result["ci"]["var"][1]-result["ci"]["var"][0],
                                 "es_interval_width": result["ci"]["es"][1]-result["ci"]["es"][0],
                                 "var_resolved": result["var_resolved"], "cost": result["resources"]["oracle_A_calls"]+result["samples"],
                                 "shots": result["resources"]["shots_total"], "runtime_ms": result["runtime_ms"],
                                 "interval_kind": result["interval_kind"]})
                    logs.append({"alpha": alpha, "budget": budget, "method": method, "seed": seed,
                                 "query_rounds": json.dumps(result["logs"], allow_nan=False)})
                pd.DataFrame(rows).to_csv(directory/"risk_results.csv", index=False)
    pd.DataFrame(logs).to_parquet(directory/"risk_rounds.parquet", compression="zstd", index=False)
    write_json(directory/"truths.json", truths)
    circuit_check(directory, table, weights, cfg)
    from .feasibility_report import make_report
    make_report(directory)
    write_json(directory/"checksums.json", {str(f.relative_to(directory)): digest(f) for f in directory.rglob("*") if f.is_file() and f.name != "checksums.json"})
    return directory

def circuit_check(directory, table, weights, cfg):
    from .tail_oracle import TailOracle, excess_payload
    from qiskit.quantum_info import Statevector
    from qiskit import qpy, transpile
    losses = table.losses(weights)
    tau = risk(losses, table.probabilities, .975)["var"]
    payload = excess_payload(losses, tau, float(np.ptp(losses)))
    oracle = TailOracle(table.probabilities, payload)
    check = oracle.validate()
    exact = float(table.probabilities@payload)
    rows = []
    for k in (0, 1, 3, 7):
        start = perf_counter()
        circuit = oracle.amplified(k)
        actual = float(Statevector.from_instruction(circuit).probabilities([oracle.n])[1])
        expected = float(np.sin((2*k+1)*np.arcsin(np.sqrt(exact)))**2)
        compiled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=0)
        rows.append({"grover_power": k, "probability_error": abs(actual-expected), "depth": compiled.depth(),
                     "cx_gates": int(compiled.count_ops().get("cx", 0)), "qubits": compiled.num_qubits,
                     "compile_and_simulate_s": perf_counter()-start})
        with (directory/f"circuit_k{k}.qpy").open("wb") as handle:
            qpy.dump([circuit, compiled], handle)
    full = estimate_payload(table.probabilities, payload, "grid_biqae", replace(cfg, backend="qiskit_statevector", max_grover_power=7), 16384, 901, epsilon=.001)
    write_json(directory/"circuit_verification.json", {"oracle": check, "amplified": rows, "actual_full_circuit_BIQAE": {
               "truth": exact, "estimate": full.estimate, "interval": list(full.interval), "resources": full.ledger.to_dict(), "rounds": full.rounds}})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "evaluate", "all"])
    parser.add_argument("--directory", default="artifacts/feasibility_trained_v1")
    args = parser.parse_args()
    directory = guarded(args.directory)
    if args.stage in {"train", "all"}:
        train(directory)
    if args.stage in {"evaluate", "all"}:
        evaluate(directory)
    say(str(directory))

if __name__ == "__main__":
    main()
