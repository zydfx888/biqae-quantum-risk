"""Discrete left quantile and Rockafellar-Uryasev ES, including atoms."""
import numpy as np

def risk(losses, probabilities, alpha):
    x, p = np.asarray(losses), np.asarray(probabilities)
    if not 0 < alpha < 1 or len(x) == 0 or x.shape != p.shape:
        raise ValueError("Invalid risk inputs")
    if not np.isfinite(x).all() or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Invalid losses/probabilities")
    order = np.argsort(x, kind="stable")
    idx = min(np.searchsorted(np.cumsum(p[order]), alpha, side="left"), len(x)-1)
    var = float(x[order[idx]])
    return {"var": var, "es": float(var + p @ np.maximum(x-var, 0)/(1-alpha))}

def cvar(table, weights, alpha, tau=None):
    losses = table.losses(weights)
    if tau is None:
        return risk(losses, table.probabilities, alpha)["es"]
    return float(tau + table.probabilities @ np.maximum(losses-tau, 0)/(1-alpha))

def tail_weights(table, weights, alpha):
    losses = table.losses(weights)
    tau = risk(losses, table.probabilities, alpha)["var"]
    above, equal = losses > tau, losses == tau
    remaining = max(0., (1-alpha)-table.probabilities[above].sum())
    fraction = min(1., remaining / table.probabilities[equal].sum())
    return above.astype(float) + fraction*equal

def gradient(table, weights, alpha):
    # Fractional mass at VaR gives a valid subgradient of the minimized CVaR.
    mass = table.probabilities * tail_weights(table, weights, alpha)
    return -(mass @ table.returns)/(1-alpha)

