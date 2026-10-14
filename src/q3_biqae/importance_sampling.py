from .es_estimation import estimate_risk as _estimate

def estimate_risk(distribution, alpha, budget, seed, **kwargs):
    return _estimate(distribution, alpha, budget, seed, method="is", **kwargs)

