import numpy as np
from scipy.optimize import linprog

def solve_cvar(table, alpha, w_max=.6):
    """LP variables: weights[d], free tau, nonnegative excess[S]."""
    scenarios, assets = table.returns.shape
    if not 0 < alpha < 1 or assets*w_max < 1:
        raise ValueError("Invalid alpha or infeasible constraints")
    c = np.r_[np.zeros(assets), 1, table.probabilities/(1-alpha)]
    inequalities = np.c_[-table.returns, -np.ones(scenarios), -np.eye(scenarios)]
    equality = np.zeros((1, len(c)))
    equality[0, :assets] = 1
    result = linprog(c, A_ub=inequalities, b_ub=np.zeros(scenarios), A_eq=equality, b_eq=[1],
                     bounds=[(0, w_max)]*assets+[(None, None)]+[(0, None)]*scenarios, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return {"weights": result.x[:assets].tolist(), "tau": float(result.x[assets]), "cvar": float(result.fun), "solver": "scipy_highs_lp"}

