from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.stats import qmc
from .statistics import mean_interval, replicate_interval
from .resource_ledger import ResourceLedger

@dataclass
class AmplitudeResult:
    estimate: float
    interval: tuple[float, float]
    interval_kind: str
    converged: bool
    ledger: ResourceLedger
    rounds: list

def estimate_payload(probabilities, payload, method, config, budget, seed, delta=None, epsilon=None):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(payload, dtype=float)
    if p.shape != y.shape or not np.isfinite(y).all() or np.any((y < 0)|(y > 1)):
        raise ValueError("Invalid amplitude payload")
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Invalid amplitude probability table")
    if budget < 1:
        raise ValueError("Empty amplitude budget")
    delta = config.failure_probability if delta is None else delta
    epsilon = config.epsilon if epsilon is None else epsilon
    if method in {"mc", "sobol", "is"}:
        start = perf_counter()
        rng = np.random.default_rng(seed)
        ledger = ResourceLedger(samples=budget)
        logs = []
        if method == "sobol" and budget >= 16:
            # Independent scrambled replicates: do not bootstrap dependent Sobol points.
            replicas = 8
            per = 2**int(np.floor(np.log2(budget//replicas)))
            means = []
            for _ in range(replicas):
                u = qmc.Sobol(1, scramble=True, seed=int(rng.integers(2**31))).random_base2(int(np.log2(per)))[:, 0]
                idx = np.minimum(np.searchsorted(np.cumsum(p), u, side="right"), len(p)-1)
                means.append(float(y[idx].mean()))
            point, ci = float(np.mean(means)), replicate_interval(means, delta)
            # In 1D every scrambled 2^m Sobol net places one point in each
            # dyadic cell. The step integrand has known bounded variation.
            # Keep this discrepancy guard even when replicate variance is zero.
            discrepancy_bound = float(np.abs(np.diff(y)).sum()/per)
            ci = (max(0., min(ci[0], point-discrepancy_bound)),
                  min(1., max(ci[1], point+discrepancy_bound)))
            ledger.samples = per*replicas
            kind = "sobol_1d_discrepancy_guard_with_replicate_t"
            logs.append({"replicate_means": means, "scrambling": True, "discrepancy_bound": discrepancy_bound})
        else:
            proposal = p
            bound = 1.
            if method == "is":
                # Fixed 50% mixture with a payload tilt; no reference value tuning.
                tilt = p*(.1+y)
                proposal = .5*p+.5*tilt/tilt.sum()
                bound = float(np.max(np.divide(p, proposal, out=np.zeros_like(p), where=proposal>0)))
            idx = rng.choice(len(p), budget, p=proposal)
            ratios = p[idx]/proposal[idx]
            values = y[idx]*ratios
            point = float(np.clip(values.mean(), 0, 1))
            ci = mean_interval(values, delta, bound)
            kind = "hoeffding_finite_sample"
            logs.append({"effective_sample_size": float(ratios.sum()**2/(ratios@ratios)),
                         "likelihood_ratio_min": float(ratios.min()), "likelihood_ratio_max": float(ratios.max())})
        ledger.runtime_ms = (perf_counter()-start)*1000
        return AmplitudeResult(point, ci, kind, (ci[1]-ci[0])/2 <= epsilon, ledger, logs)
    from .measurement import Measurement
    measurement = Measurement(p, y, config, seed)
    if method == "iqae":
        from .iqae import estimate
    elif method == "mlae":
        from .mlae import estimate
    elif method in {"grid_biqae", "normal_biqae", "beta_biqae"}:
        from .biqae import estimate
    else:
        raise ValueError(f"Unknown method {method}")
    return estimate(measurement, config, budget, delta, epsilon, method)
