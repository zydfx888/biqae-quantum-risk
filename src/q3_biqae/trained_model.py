"""Small Gaussian-mixture risk model, fitted by EM using training rows only."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from .config import guarded
from .scenario_table import ScenarioTable

@dataclass
class TrainedRiskModel:
    mixture: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def log_density(self, returns):
        z = (np.asarray(returns)-self.center)/self.scale
        logs = np.column_stack([np.log(p)+multivariate_normal.logpdf(z, m, c)
                                for p, m, c in zip(self.mixture, self.means, self.covariances)])
        return logsumexp(logs, axis=1)-np.log(self.scale).sum()

    def sample(self, count, seed):
        rng = np.random.default_rng(seed)
        labels = rng.choice(len(self.mixture), count, p=self.mixture)
        z = np.empty((count, len(self.center)))
        for k in range(len(self.mixture)):
            mask = labels == k
            z[mask] = rng.multivariate_normal(self.means[k], self.covariances[k], mask.sum())
        return self.center+z*self.scale

    def save(self, path):
        np.savez_compressed(guarded(path), mixture=self.mixture, means=self.means,
                            covariances=self.covariances, center=self.center, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: data[name] for name in ("mixture", "means", "covariances", "center", "scale")})

def train_mixture(returns, components=2, seed=13, iterations=200):
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or len(x) < 20 or not np.isfinite(x).all():
        raise ValueError("Invalid training returns")
    center, scale = x.mean(axis=0), x.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Constant asset in training data")
    z = (x-center)/scale
    rng = np.random.default_rng(seed)
    mix = np.full(components, 1/components)
    means = z[rng.choice(len(z), components, replace=False)].copy()
    cov = np.array([np.cov(z.T)*(0.5+1.5*k/max(1, components-1)) for k in range(components)])
    history = []
    for step in range(iterations):
        logs = np.column_stack([np.log(p)+multivariate_normal.logpdf(z, m, c)
                                for p, m, c in zip(mix, means, cov)])
        normalizer = logsumexp(logs, axis=1)
        objective = float(normalizer.mean())
        history.append({"iteration": step, "training_average_log_likelihood_standardized": objective})
        responsibility = np.exp(logs-normalizer[:, None])
        mass = responsibility.sum(axis=0)
        mix = mass/len(z)
        means = responsibility.T @ z/mass[:, None]
        for k in range(components):
            diff = z-means[k]
            cov[k] = (diff.T*responsibility[:, k]) @ diff/mass[k]+1e-6*np.eye(z.shape[1])
        if step > 10 and abs(history[-1]["training_average_log_likelihood_standardized"]-history[-2]["training_average_log_likelihood_standardized"]) < 1e-7:
            break
    return TrainedRiskModel(mix, means, cov, center, scale), history

def freeze_model_scenarios(returns, states=16):
    """Tail bins learned from generated returns; empirical mass, no alpha labels."""
    weights = np.full(returns.shape[1], 1/returns.shape[1])
    order = np.argsort(-returns@weights, kind="stable")
    # Unequal body/tail division avoids putting both test alphas exactly at
    # a predefined atom boundary. Fixed before any estimation results exist.
    body = states//2+1
    edges = np.r_[np.linspace(0, .9, body+1), np.linspace(.9, 1, states-body+1)[1:]]
    cuts = np.rint(edges*len(returns)).astype(int)
    centers, mass = [], []
    for start, end in zip(cuts[:-1], cuts[1:]):
        idx = order[start:end]
        centers.append(returns[idx].mean(axis=0))
        mass.append(len(idx)/len(returns))
    return ScenarioTable(centers, mass)
