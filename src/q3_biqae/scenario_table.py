from dataclasses import dataclass
import hashlib
import numpy as np
from .config import guarded

@dataclass(frozen=True)
class ScenarioTable:
    returns: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self):
        r, p = np.array(self.returns, dtype=float, copy=True), np.array(self.probabilities, dtype=float, copy=True)
        if r.ndim != 2 or p.shape != (len(r),) or len(r) == 0:
            raise ValueError("Expected returns[S,d], probabilities[S]")
        if not np.isfinite(r).all() or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("Nonfinite values or negative probabilities")
        if not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError("Probability mass must sum to one")
        r.setflags(write=False)
        p.setflags(write=False)
        object.__setattr__(self, "returns", r)
        object.__setattr__(self, "probabilities", p)

    def losses(self, weights):
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.returns.shape[1],) or not np.isfinite(w).all():
            raise ValueError("Invalid weights")
        return -self.returns @ w

    @property
    def sha256(self):
        h = hashlib.sha256()
        for a in (self.returns, self.probabilities):
            h.update(str(a.shape).encode())
            h.update(a.astype("<f8").tobytes())
        return h.hexdigest()

    def save(self, path):
        path = guarded(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, returns=self.returns, probabilities=self.probabilities)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["returns"], data["probabilities"])

