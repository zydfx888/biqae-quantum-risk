from time import perf_counter
import numpy as np
from .cvar_gradient import joint_gradient
from .exact_reference import cvar
from .resource_ledger import ResourceLedger

def project_weights(values, maximum):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or maximum*len(values) < 1:
        raise ValueError("Invalid projection input")
    low, high = float(values.min()-maximum), float(values.max())
    for _ in range(100):
        middle = (low+high)/2
        if np.clip(values-middle, 0, maximum).sum() > 1:
            low = middle
        else:
            high = middle
    return np.clip(values-(low+high)/2, 0, maximum)

def optimize(table, config, method="exact", seed=13, test_table=None):
    start = perf_counter()
    w = np.full(table.returns.shape[1], 1/table.returns.shape[1])
    tau = 0.
    # Precondition tau using a frozen return scale; all methods share this rule.
    scale = max(float(np.max(np.abs(table.returns))), 1e-6)
    ledger, history, logs = ResourceLedger(), [], []
    average = np.zeros_like(w)
    previous = w.copy()
    for step in range(config.optimization_steps):
        grad, grad_tau, cost, rounds = joint_gradient(table, w, tau, config.alpha, method, config, config.budget, seed+step)
        ledger.merge(cost)
        eta = config.learning_rate/np.sqrt(step+1)
        w = project_weights(w-eta*grad/scale, config.w_max)
        tau = float(np.clip(tau-eta*scale*grad_tau, -scale, scale))
        average += w
        averaged = average/(step+1)
        history.append({"step": step, "weights": w.tolist(), "average_weights": averaged.tolist(), "tau": tau,
                        "cvar": cvar(table, averaged, config.alpha), "joint_objective": cvar(table, w, config.alpha, tau),
                        "turnover": float(np.abs(w-previous).sum()), "oracle_A_calls": ledger.oracle_A_calls,
                        "samples": ledger.samples})
        logs.append({"step": step, "queries": rounds})
        previous = w.copy()
    final = average/config.optimization_steps
    result = {"method": method, "seed": seed, "weights": final.tolist(), "cvar": cvar(table, final, config.alpha),
              "constraint_violation": float(abs(final.sum()-1)+np.maximum(-final, 0).sum()+np.maximum(final-config.w_max, 0).sum()),
              "resources": ledger.to_dict(), "runtime_ms": (perf_counter()-start)*1000, "history": history, "logs": logs}
    if test_table is not None:
        result["out_of_sample_es"] = cvar(test_table, final, config.alpha)
    return result
