import numpy as np
from .amplitude import estimate_payload
from .resource_ledger import ResourceLedger

def joint_gradient(table, weights, tau, alpha, method, config, budget, seed):
    """Joint RU subgradient; strict > chooses a valid subgradient at atoms.

    Unlike the minimized-CVaR reference, this updates tau as a variable and
    therefore needs no exact quantile inside the estimator-driven optimizer.
    """
    active = (table.losses(weights) > tau).astype(float)
    if method == "exact":
        mass = table.probabilities*active
        return -(mass @ table.returns)/(1-alpha), float(1-mass.sum()/(1-alpha)), ResourceLedger(), []
    if method in {"mc", "sobol", "is"}:
        from .classical_gradient import shared_batch_gradient
        return shared_batch_gradient(table, active, alpha, budget, seed, method)
    assets = len(weights)
    queries = 2*assets+1
    allocation = budget//queries
    if allocation < 1:
        raise ValueError("Gradient budget must cover all signed payload queries")
    ledger, logs = ResourceLedger(), []
    components = []
    rng = np.random.default_rng(seed)
    for i in range(assets):
        values = []
        for sign in (1, -1):
            returns = np.maximum(sign*table.returns[:, i], 0)
            bound = float(returns.max())
            if bound == 0:
                values.append(0.)
                continue
            result = estimate_payload(table.probabilities, returns*active/bound, method, config, allocation,
                                      int(rng.integers(2**31)), config.failure_probability/queries)
            ledger.merge(result.ledger)
            logs.append({"asset": i, "sign": sign, "bound": bound, "rounds": result.rounds})
            values.append(bound*result.estimate)
        components.append(-(values[0]-values[1])/(1-alpha))
    probability = estimate_payload(table.probabilities, active, method, config, allocation,
                                   int(rng.integers(2**31)), config.failure_probability/queries)
    ledger.merge(probability.ledger)
    logs.append({"component": "tau", "rounds": probability.rounds})
    return np.array(components), 1-probability.estimate/(1-alpha), ledger, logs
