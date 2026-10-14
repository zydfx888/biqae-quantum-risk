import numpy as np
from .amplitude import estimate_payload
from .resource_ledger import ResourceLedger

def search_var(losses, probabilities, alpha, method, config, budget, seed, delta=None):
    levels = np.unique(losses)
    low, high = 0, len(levels)-1
    ledger, logs = ResourceLedger(), []
    remaining = budget
    rng = np.random.default_rng(seed)
    queries = max(1, int(np.ceil(np.log2(len(levels)))))
    max_queries = 2*queries+2
    delta = config.failure_probability if delta is None else delta
    epsilon = config.epsilon
    queried = set()
    for query in range(max_queries):
        if low == high or remaining < 1:
            break
        mid = (low+high)//2
        # At F(x)==alpha no finite precision test can certify equality.
        # Probe neighbouring levels to narrow the quantile bracket instead
        # of wasting every remaining query at this single ambiguous point.
        available = [i for i in range(low, high) if i not in queried]
        if mid in queried and available:
            mid = min(available, key=lambda i: abs(i-mid))
        queried.add(mid)
        allocation = min(remaining, max(1, budget//max_queries)*2**(query//queries))
        result = estimate_payload(probabilities, (losses <= levels[mid]).astype(float), method, config, allocation,
                                  int(rng.integers(2**31)), delta/max_queries, epsilon)
        ledger.merge(result.ledger)
        cost = result.ledger.samples+result.ledger.oracle_A_calls
        remaining -= cost
        logs.append({"threshold": float(levels[mid]), "cdf_estimate": result.estimate,
                     "cdf_interval": list(result.interval), "interval_kind": result.interval_kind,
                     "rounds": result.rounds, "stage": "var", "query": query})
        if result.interval[0] >= alpha:
            high = mid
        elif result.interval[1] < alpha:
            low = mid+1
        else:
            # Never collapse an ambiguous interval based on the point estimate.
            epsilon /= 2
    return {"var": float(levels[(low+high)//2]), "interval": [float(levels[low]), float(levels[high])],
            "resolved": low == high, "remaining_budget": remaining, "logs": logs, "ledger": ledger}
