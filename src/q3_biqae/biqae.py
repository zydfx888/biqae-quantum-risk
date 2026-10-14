"""Grid BIQAE plus explicitly approximate normal/beta moment projections.

These are the scheme's engineering variants, not a verbatim reproduction of
the paper's hybrid base-3/base-5 schedules or its complexity guarantee.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, norm
from .amplitude import AmplitudeResult
from .iqae import safe_power

def grid_interval(grid, posterior, delta):
    cdf = np.cumsum(posterior)
    low, high = np.searchsorted(cdf, [delta/2, 1-delta/2])
    # Include adjacent cells to expose finite grid resolution in intervals.
    return np.array([grid[max(0, low-1)], grid[min(len(grid)-1, high+1)]])

def estimate(measurement, config, budget, delta, epsilon, method="grid_biqae"):
    # Uniform theta mesh, amplitude-beta prior with the correct Jacobian.
    theta = (np.arange(config.grid_size)+.5)/config.grid_size*np.pi/2
    amplitudes = np.sin(theta)**2
    logp = beta.logpdf(amplitudes, *config.prior)+np.log(np.sin(2*theta))
    logp -= logsumexp(logp)
    credible_theta = np.array([0., np.pi/2])
    remaining = budget
    for _ in range(config.max_rounds):
        k, branch = safe_power(credible_theta, min(config.max_grover_power, max(0, (remaining-1)//2)))
        shots = min(config.shots_per_round, remaining//(2*k+1))
        if shots == 0:
            break
        h = measurement.sample(k, shots)
        remaining -= shots*(2*k+1)
        prob = np.clip(np.sin((2*k+1)*theta)**2, 1e-15, 1-1e-15)
        logp += h*np.log(prob)+(shots-h)*np.log1p(-prob)
        logp -= logsumexp(logp)
        posterior = np.exp(logp)
        if method == "normal_biqae":
            mean = posterior @ theta
            sd = max(np.sqrt(posterior @ (theta-mean)**2), np.pi/(2*config.grid_size))
            logp = norm.logpdf(theta, mean, sd)
        elif method == "beta_biqae":
            mean = float(posterior @ amplitudes)
            variance = max(float(posterior @ (amplitudes-mean)**2), 1e-14)
            concentration = max(mean*(1-mean)/variance-1, .01)
            logp = beta.logpdf(amplitudes, max(mean*concentration, .01), max((1-mean)*concentration, .01))+np.log(np.sin(2*theta))
        logp -= logsumexp(logp)
        posterior = np.exp(logp)
        credible_theta = grid_interval(theta, posterior, delta/config.max_rounds)
        ci = np.sin(credible_theta)**2
        measurement.rounds[-1].update(theta_interval=credible_theta.tolist(), amplitude_interval=ci.tolist(), branch=branch,
                                      grid_size=config.grid_size,
                                      posterior_entropy=float(-np.sum(posterior*np.log(np.maximum(posterior, 1e-300)))))
        if config.record_posterior:
            measurement.rounds[-1]["posterior"] = posterior.tolist()
        if config.stopping_rule == "interval" and (ci[1]-ci[0])/2 <= epsilon:
            break
    posterior = np.exp(logp)
    ci = np.sin(credible_theta)**2
    # Boundary cells represent the full closed parameter domain.
    if credible_theta[0] == theta[0]:
        ci[0] = 0.
    if credible_theta[1] == theta[-1]:
        ci[1] = 1.
    measurement.finalize_resources()
    return AmplitudeResult(float(posterior @ amplitudes), tuple(map(float, ci)), "bayesian_credible_grid" if method == "grid_biqae" else "bayesian_credible_moment_approximate",
                           (ci[1]-ci[0])/2 <= epsilon, measurement.ledger, measurement.rounds)
