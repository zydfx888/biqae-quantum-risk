from time import perf_counter
import numpy as np
from .amplitude import estimate_payload
from .tail_oracle import excess_payload
from .var_search import search_var

def estimate_risk(distribution, alpha, budget, seed, method="iqae", config=None, weights=None):
    from .config import Config
    config = Config() if config is None else config
    started = perf_counter()
    weights = np.full(distribution.returns.shape[1], 1/distribution.returns.shape[1]) if weights is None else weights
    losses, p = distribution.losses(weights), distribution.probabilities
    var_budget = max(1, int(budget*config.var_budget_fraction))
    search = search_var(losses, p, alpha, method, config, var_budget, seed, config.failure_probability/2)
    tau = search["var"]
    # Frozen range is selected from the complete input support before sampling.
    bound = float(np.ptp(losses))
    tail = estimate_payload(p, excess_payload(losses, tau, bound), method, config, budget-var_budget,
                            seed+1000003, config.failure_probability/2,
                            min(.5, config.epsilon*(1-alpha)/max(bound, 1e-12)))
    point = tau+bound*tail.estimate/(1-alpha)
    lo, hi = [tau+bound*x/(1-alpha) for x in tail.interval]
    # f(tau) >= min f; Lipschitz propagation covers unresolved VaR atoms.
    distance = max(abs(tau-x) for x in search["interval"])
    lo -= max(1., alpha/(1-alpha))*distance
    ci = [max(float(losses.min()), lo), min(float(losses.max()), hi)]
    if ci[0] > ci[1]:
        ci = [float(losses.min()), float(losses.max())]
    ledger = search["ledger"].merge(tail.ledger)
    logs = search["logs"]+[{'stage': 'es', 'threshold': tau, 'rounds': tail.rounds,
                             'amplitude_interval': list(tail.interval), 'interval_kind': tail.interval_kind}]
    return {"var": tau, "es": float(np.clip(point, losses.min(), losses.max())),
            "ci": {"var": search["interval"], "es": ci}, "var_resolved": search["resolved"],
            "converged": search["resolved"] and (ci[1]-ci[0])/2 <= config.epsilon,
            "samples": ledger.samples, "runtime_ms": (perf_counter()-started)*1000,
            "resources": ledger.to_dict(), "logs": logs, "backend": config.backend if method not in {"mc", "sobol", "is"} else "classical",
            "interval_kind": tail.interval_kind, "seed": seed, "method": method,
            "budget": budget, "es_threshold_uncertainty_bound": max(1., alpha/(1-alpha))*distance}

