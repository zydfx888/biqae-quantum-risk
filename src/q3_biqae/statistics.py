import numpy as np
from scipy.stats import beta, t, wilcoxon

def clopper_pearson(h, n, delta):
    return (0. if h == 0 else float(beta.ppf(delta/2, h, n-h+1)),
            1. if h == n else float(beta.ppf(1-delta/2, h+1, n-h)))

def mean_interval(values, delta, bound=1.):
    values = np.asarray(values)
    width = bound*np.sqrt(np.log(2/delta)/(2*len(values)))
    return max(0., float(values.mean()-width)), min(1., float(values.mean()+width))

def replicate_interval(means, delta):
    means = np.asarray(means)
    width = t.ppf(1-delta/2, len(means)-1)*np.std(means, ddof=1)/np.sqrt(len(means))
    return max(0., float(means.mean()-width)), min(1., float(means.mean()+width))

def bootstrap_interval(values, seed=13, repetitions=2000, delta=.05):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).choice(values, (repetitions, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [delta/2, 1-delta/2]).tolist()

def paired_test(first, second):
    diff = np.asarray(first)-np.asarray(second)
    return 1. if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)

def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()

