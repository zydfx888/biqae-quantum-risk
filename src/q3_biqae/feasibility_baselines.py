"""Strong classical baselines: reuse one sample set for both VaR and ES."""
from time import perf_counter
import numpy as np
from scipy.stats import qmc
from .exact_reference import risk

def estimate_classical_risk(table, weights, alpha, budget, seed, method, delta=.05):
    start = perf_counter()
    losses = table.losses(weights)
    order = np.argsort(losses, kind="stable")
    losses, p = losses[order], table.probabilities[order]
    rng = np.random.default_rng(seed)
    count = budget
    q = p.copy()
    if method == "sobol":
        power = int(np.floor(np.log2(budget)))
        count = 2**power
        u = qmc.Sobol(1, scramble=True, seed=seed).random_base2(power)[:, 0]
        indices = np.minimum(np.searchsorted(np.cumsum(p), u, side="right"), len(p)-1)
        band = 1/count  # 1D scrambled dyadic net discrepancy; sorted inverse CDF.
        kind = "one_dimensional_sobol_discrepancy"
    else:
        if method == "is":
            score = (losses-losses.min())/max(float(np.ptp(losses)), 1e-12)
            tilt = p*(.05+score**4)
            q = .5*p+.5*tilt/tilt.sum()
        indices = rng.choice(len(p), count, p=q)
        band = np.sqrt(np.log(2/delta)/(2*count))
        kind = "DKW_uniform_CDF" if method == "mc" else "self_normalized_IS_Hoeffding_union"
    counts = np.bincount(indices, minlength=len(p))
    ratios = np.divide(p, q, out=np.zeros_like(p), where=q>0)
    mass = counts*ratios/count
    denominator = mass.sum()
    empirical = mass/denominator
    cdf = np.cumsum(empirical)
    if method == "is":
        width = float(ratios.max()*np.sqrt(np.log(2*(len(p)+1)/delta)/(2*count)))
        lower = np.maximum(0, np.cumsum(mass)-width)/(denominator+width)
        upper = np.minimum(1, (np.cumsum(mass)+width)/max(denominator-width, 1e-12))
        band = float(max(np.max(cdf-lower), np.max(upper-cdf)))
    point = risk(losses, empirical, alpha)
    lower_index = min(np.searchsorted(cdf, max(0., alpha-band)), len(losses)-1)
    # A band reaching one cannot exclude unsampled upper-support states.
    # Empirical quantile(1) may be below the known maximum when a tail atom
    # was not sampled, so using it here would invalidate interval coverage.
    upper_index = len(losses)-1 if alpha+band >= 1 else min(np.searchsorted(cdf, alpha+band), len(losses)-1)
    es_width = float(np.ptp(losses))*band/(1-alpha)
    var_interval = [float(losses[lower_index]), float(losses[upper_index])]
    es_interval = [max(float(losses.min()), point["es"]-es_width), min(float(losses.max()), point["es"]+es_width)]
    return {**point, "ci": {"var": var_interval, "es": es_interval}, "samples": count,
            "runtime_ms": (perf_counter()-start)*1000, "resources": {"oracle_A_calls": 0, "shots_total": 0},
            "var_resolved": lower_index==upper_index, "interval_kind": kind,
            "logs": [{"method": method, "state_counts": counts.tolist(), "likelihood_ratios": ratios.tolist(),
                      "empirical_probabilities": empirical.tolist(), "cdf_uniform_error_bound": band,
                      "effective_sample_size": float((counts@ratios)**2/(counts@(ratios**2)))}]}
