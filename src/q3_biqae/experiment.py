from datetime import datetime, timezone
from dataclasses import replace
from time import perf_counter
import importlib.metadata
import platform
import sys
import uuid
import json
import numpy as np
from .config import ROOT, guarded, digest, write_json
from .distributions import generate
from .discretization import discretize
from .scenario_table import ScenarioTable
from .exact_reference import risk
from .es_estimation import estimate_risk
from .classical_cvar import solve_cvar
from .optimizer import optimize
from .reporting import save_estimates, plot_results, report

def prepare(config):
    weights = np.full(config.assets, 1/config.assets)
    if config.dataset == "french":
        from .data_prepare import parse_french, split_asof
        if not config.input_file or not config.train_end or not config.test_start:
            raise ValueError("French input requires input_file, train_end and test_start")
        train, test = split_asof(parse_french(guarded(config.input_file)), config.train_end, config.test_start)
        raw = ScenarioTable(train.iloc[:, :config.assets].values, np.full(len(train), 1/len(train)))
        heldout = ScenarioTable(test.iloc[:, :config.assets].values, np.full(len(test), 1/len(test)))
    elif config.dataset == "external":
        if not config.input_file or not guarded(config.input_file).is_relative_to(guarded("data/external_frozen")):
            raise ValueError("External data must be copied into data/external_frozen via import-frozen")
        raw = ScenarioTable.load(guarded(config.input_file))
        return raw, None, {"scope": "external_interface_extension_only"}, weights, raw
    else:
        raw = generate(config.dataset, config.scenarios, config.assets, config.distribution_seed, config.rho, config.df)
        heldout = generate(config.dataset, config.scenarios, config.assets, config.distribution_seed+1, config.rho, config.df)
    reduced, errors = discretize(raw, weights, config.states, config.alpha, config.discretization)
    return reduced, heldout, errors, weights, raw

def circuit_audit(directory, table, weights, truth):
    from qiskit import qpy, transpile
    from .tail_oracle import TailOracle, excess_payload
    losses = table.losses(weights)
    circuits = guarded(directory/"circuits")
    circuits.mkdir()
    rows = []
    for name, payload in {"tail": (losses >= truth["var"]).astype(float),
                          "excess": excess_payload(losses, truth["var"], float(np.ptp(losses)))}.items():
        tick = perf_counter()
        oracle = TailOracle(table.probabilities, payload)
        validation = oracle.validate()
        build_ms = (perf_counter()-tick)*1000
        tick = perf_counter()
        compiled = transpile(oracle.circuit, basis_gates=["u", "cx"], seed_transpiler=0, optimization_level=1)
        compilation_ms = (perf_counter()-tick)*1000
        with (circuits/f"{name}.qpy").open("wb") as handle:
            qpy.dump([oracle.circuit, compiled], handle)
        rows.append({"oracle": name, **validation, "qubits": compiled.num_qubits,
                     "depth": compiled.depth(), "cx_gates": int(compiled.count_ops().get("cx", 0)),
                     "state_preparation_and_validation_ms": build_ms, "compilation_ms": compilation_ms,
                     "table_entries": len(table.probabilities), "implementation": "exact probability loading + fused table controlled Ry; no loss register in fused path"})
    write_json(circuits/"audit.json", rows)

def run(config, with_optimization=False, progress=print):
    tick = perf_counter()
    run_id = "biqae_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]
    directory = guarded(guarded(config.output)/run_id)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/"config.json", config.to_dict())
    packages = ["numpy", "scipy", "qiskit", "qiskit-aer", "qiskit-algorithms", "pandas", "pyarrow", "matplotlib"]
    write_json(directory/"environment.json", {"python": sys.version, "platform": platform.platform(),
               "packages": {name: importlib.metadata.version(name) for name in packages},
               "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (ROOT/"src/q3_biqae").glob("*.py")}})
    write_json(directory/"protocol.json", {"version": 1, "quantile": "inf{x:F(x)>=alpha}", "es": "Rockafellar-Uryasev",
               "A_calls": "sum(shots*(2*k+1)), includes A dagger", "budget_unit": "A/A-dagger calls for quantum; distribution samples for classical",
               "var_budget_fraction": config.var_budget_fraction, "backends": {"analytic_shots": "classical ideal likelihood simulator", "qiskit_statevector": "full ideal circuit with finite shots", "aer_shots": "ideal Aer circuit shots", "aer_noise": "Aer depolarizing circuit noise"}})
    try:
        table, test_table, errors, weights, raw = prepare(config)
        table.save(directory/"distribution.npz")
        raw.save(directory/"raw_scenarios.npz")
        truth = risk(table.losses(weights), table.probabilities, config.alpha)
        write_json(directory/"truth.json", truth)
        write_json(directory/"discretization.json", errors)
        write_json(directory/"data_manifest.json", {"dataset_id": config.dataset, "generator_version": "q3_biqae/0.1.0",
                   "random_seed": config.distribution_seed, "state_count": len(table.probabilities), "weights": weights.tolist(),
                   "sha256": digest(directory/"distribution.npz"), "table_sha256": table.sha256,
                   "raw_sha256": raw.sha256, "train_end": config.train_end, "test_start": config.test_start,
                   "input_file_sha256": digest(guarded(config.input_file)) if config.input_file else None})
        circuit_audit(directory, table, weights, truth)
        estimates = []
        for method in config.methods:
            for seed in config.seeds:
                progress(f"[{run_id}] risk {method} seed={seed}")
                result = estimate_risk(table, config.alpha, config.budget, seed, method, config, weights)
                estimates.append(result)
                # Checkpoint every run; do not drop failed or unresolved seeds.
                checkpoint = directory/"checkpoints"/f"{method}_{seed}"
                checkpoint.mkdir(parents=True)
                save_estimates(checkpoint, [result], truth)
        frame = save_estimates(directory, estimates, truth)
        optimizations = []
        if with_optimization:
            reference = solve_cvar(table, config.alpha, config.w_max)
            write_json(directory/"optimization_reference.json", reference)
            for method in ["exact", *[m for m in config.methods if m in {"mc", "iqae", "grid_biqae", "normal_biqae", "beta_biqae"}]]:
                progress(f"[{run_id}] optimization {method}")
                result = optimize(table, config, method, config.seeds[0], test_table)
                result["optimality_gap"] = result["cvar"]-reference["cvar"]
                result["weight_l1_distance"] = float(np.abs(np.array(result["weights"])-reference["weights"]).sum())
                from .reporting import save_optimization
                save_optimization(directory, result)
                optimizations.append(result)
        plot_results(directory, frame, optimizations)
        report(directory, config, truth, frame, optimizations)
        write_json(directory/"status.json", {"status": "complete", "elapsed_ms": (perf_counter()-tick)*1000,
                   "run_id": run_id, "converged_count": int(frame.converged.sum()), "estimate_count": len(frame),
                   "acceptance": "execution_only; coverage/query advantage require formal sweep"})
    except Exception as exc:
        write_json(directory/"status.json", {"status": "failed", "error": repr(exc), "elapsed_ms": (perf_counter()-tick)*1000})
        raise
    write_json(directory/"checksums.json", {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()})
    return directory

def sweep(config, grid, progress=print):
    from itertools import product
    paths = []
    for dataset, states, alpha, budget in product(grid["datasets"], grid["states"], grid["alphas"], grid["budgets"]):
        current = replace(config, dataset=dataset, states=states, alpha=alpha, budget=budget,
                          seeds=grid.get("seeds", config.seeds))
        paths.append(str(run(current, progress=progress)))
    write_json(guarded(config.output)/"last_sweep.json", {"runs": paths, "grid": grid})
    return paths
