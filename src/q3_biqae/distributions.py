"""Self-contained correlated generators; all randomness is explicitly seeded."""
import numpy as np
from scipy.stats import t, norm
from .scenario_table import ScenarioTable

def generate(kind, n, assets, seed, rho=.5, df=5):
    rng = np.random.default_rng(seed)
    correlation = np.full((assets, assets), rho)
    np.fill_diagonal(correlation, 1)
    z = rng.standard_normal((n, assets)) @ np.linalg.cholesky(correlation).T
    scale = np.linspace(.008, .016, assets)
    mean = np.linspace(.0001, .0005, assets)
    kind = kind.removeprefix("synthetic_")
    if kind in {"student_t", "t_copula"}:
        z /= np.sqrt(rng.chisquare(df, (n, 1))/df)
        if kind == "t_copula":
            z = norm.ppf(np.clip(t.cdf(z, df), 1e-10, 1-1e-10))
            z = np.where(z < 0, z * 1.5, z * .8)
        else:
            z *= np.sqrt((df-2)/df)
    elif kind == "regime":
        stress = rng.random(n) < .1
        z[stress] = z[stress] * 2.5 - 2
    elif kind == "extreme":
        z[rng.random(n) < .02] -= 6
    elif kind != "normal":
        raise ValueError(f"Unknown distribution: {kind}")
    return ScenarioTable(mean + z*scale, np.full(n, 1/n))

