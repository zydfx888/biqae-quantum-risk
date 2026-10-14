import numpy as np
from .scenario_table import ScenarioTable
from .exact_reference import risk, gradient

def discretize(table, weights, states, alpha, method="tail"):
    if not 2 <= states <= len(table.probabilities):
        raise ValueError("Invalid number of states")
    order = np.argsort(table.losses(weights), kind="stable")
    p, r = table.probabilities[order], table.returns[order]
    if method == "tail":
        body = max(1, states//2)
        edges = np.r_[np.linspace(0, .95, body+1), np.linspace(.95, 1, states-body+1)[1:]]
    elif method == "uniform":
        edges = np.linspace(0, 1, states+1)
    else:
        raise ValueError("Unknown binning method")
    # Split boundary atoms by probability mass, preserving every return contribution.
    right = np.cumsum(p)
    left = right-p
    probs, returns = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        overlap = np.maximum(0, np.minimum(right, hi)-np.maximum(left, lo))
        mass = overlap.sum()
        if mass > 0:
            probs.append(mass)
            returns.append(overlap @ r / mass)
    reduced = ScenarioTable(returns, np.array(probs)/sum(probs))
    raw_truth = risk(table.losses(weights), table.probabilities, alpha)
    truth = risk(reduced.losses(weights), reduced.probabilities, alpha)
    error = {f"{k}_discretization_error": truth[k]-raw_truth[k] for k in truth}
    error.update(raw_truth=raw_truth, discrete_truth=truth, bin_edges=edges.tolist(),
                 gradient_l2_error=float(np.linalg.norm(gradient(table, weights, alpha)-gradient(reduced, weights, alpha))),
                 tail_mass_error=float(reduced.probabilities[reduced.losses(weights)>=raw_truth['var']].sum()-table.probabilities[table.losses(weights)>=raw_truth['var']].sum()))
    return reduced, error

