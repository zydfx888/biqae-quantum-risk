"""Multimodal grid maximum likelihood; profile interval is asymptotic."""
import numpy as np
from scipy.stats import chi2
from .amplitude import AmplitudeResult

def estimate(measurement, config, budget, delta, epsilon, method="mlae"):
    theta = np.linspace(0, np.pi/2, config.grid_size)
    likelihood = np.zeros(len(theta))
    schedule = [0]+[2**i for i in range(config.max_grover_power.bit_length()) if 2**i <= config.max_grover_power]
    remaining = budget
    for i, k in enumerate(schedule):
        shots = min(config.shots_per_round, remaining//(2*k+1),
                    max(1, remaining//((len(schedule)-i)*(2*k+1))))
        if shots == 0:
            continue
        h = measurement.sample(k, shots)
        remaining -= shots*(2*k+1)
        p = np.clip(np.sin((2*k+1)*theta)**2, 1e-15, 1-1e-15)
        likelihood += h*np.log(p)+(shots-h)*np.log1p(-p)
    accepted = likelihood >= likelihood.max()-chi2.ppf(1-delta, 1)/2
    indices = np.flatnonzero(accepted)
    lo = max(0, indices[0]-1)
    hi = min(len(theta)-1, indices[-1]+1)
    ci = tuple(map(float, np.sin(theta[[lo, hi]])**2))
    modes = int(1+np.sum(np.diff(indices)>1))
    measurement.rounds[-1].update(amplitude_interval=list(ci), profile_components=modes)
    measurement.finalize_resources()
    return AmplitudeResult(float(np.sin(theta[np.argmax(likelihood)])**2), ci, "profile_likelihood_asymptotic",
                           (ci[1]-ci[0])/2 <= epsilon, measurement.ledger, measurement.rounds)
