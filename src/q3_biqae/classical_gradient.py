"""Estimate the entire CVaR gradient from one batch of return vectors."""
from time import perf_counter
import numpy as np
from scipy.stats import qmc
from .resource_ledger import ResourceLedger


def shared_batch_gradient(table, tail, alpha, budget, seed, method):
    """Return (weight gradient, tau gradient, ledger, logs).

    ``tail`` may be a strict indicator for joint RU optimization or include
    fractional VaR-atom mass for minimized CVaR. A scenario query retrieves
    the complete return vector, so the same batch serves every component.
    IS uses one fixed-rule vector-norm proposal and ordinary p/q correction
    (no self normalization). QMC uses one scrambled one-dimensional net.
    """
    started = perf_counter()
    p = table.probabilities
    tail = np.asarray(tail, dtype=float)
    if tail.shape != p.shape or not np.isfinite(tail).all() or np.any((tail < 0)|(tail > 1)):
        raise ValueError("Tail payload must be finite and in [0,1]")
    if not 0 < alpha < 1 or budget < 1 or int(budget) != budget:
        raise ValueError("Invalid alpha or sample budget")
    if method not in {"mc", "sobol", "is"}:
        raise ValueError("Expected classical gradient method")
    count = int(budget)
    rng = np.random.default_rng(seed)
    proposal = p.copy()
    if method == "is":
        # Common envelope for ALL signed components; no component-specific
        # proposal or extra scenario batches, and no test-set tuning.
        envelope = np.linalg.norm(table.returns*tail[:, None], axis=1)
        maximum = float(envelope.max())
        if maximum > 0:
            tilt = p*(.1+envelope/maximum)
            proposal = .5*p+.5*tilt/tilt.sum()
    if method == "sobol":
        power = count.bit_length()-1
        count = 2**power
        uniforms = qmc.Sobol(1, scramble=True, seed=seed).random_base2(power)[:, 0]
        indices = np.minimum(np.searchsorted(np.cumsum(p), uniforms, side="right"), len(p)-1)
    else:
        indices = rng.choice(len(p), size=count, p=proposal)
    ratios = p[indices]/proposal[indices]
    weighted_tail = tail[indices]*ratios
    # Signed vector arithmetic is free of the probability-amplitude constraint.
    g = -np.mean(table.returns[indices]*weighted_tail[:, None], axis=0)/(1-alpha)
    gt = float(1-weighted_tail.mean()/(1-alpha))
    ledger = ResourceLedger(samples=count, runtime_ms=(perf_counter()-started)*1000)
    counts = np.bincount(indices, minlength=len(p))
    details = {"sampling": "one_shared_return_vector_batch", "method": method,
               "seed": int(seed), "requested_budget": int(budget), "samples": count,
               "assets": table.returns.shape[1], "state_counts": counts.tolist(),
               "proposal_probabilities": proposal.tolist(), "tail_payload": tail.tolist(),
               "gradient": g.tolist(), "tau_gradient": gt,
               "effective_sample_size": float(ratios.sum()**2/(ratios@ratios)),
               "scrambling": method == "sobol"}
    return g, gt, ledger, [{"component": "all_weights_and_tau", "rounds": [details]}]
