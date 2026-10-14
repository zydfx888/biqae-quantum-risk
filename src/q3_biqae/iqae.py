"""Bounded IQAE: exact CP intervals, monotone branches, alpha spending."""
import numpy as np
from .amplitude import AmplitudeResult
from .statistics import clopper_pearson

def safe_power(interval, maximum):
    lo, hi = interval
    for k in range(maximum, -1, -1):
        factor = 2*k+1
        branch = int(np.floor((factor*lo+1e-14)/(np.pi/2)))
        if factor*hi <= (branch+1)*np.pi/2 + 1e-14:
            return k, branch
    return 0, 0

def invert_interval(interval, k, branch):
    lo, hi = np.arcsin(np.sqrt(np.clip(interval, 0, 1)))
    if branch % 2:
        result = ((branch+1)*np.pi/2-hi, (branch+1)*np.pi/2-lo)
    else:
        result = (branch*np.pi/2+lo, branch*np.pi/2+hi)
    return np.asarray(result)/(2*k+1)

def estimate(measurement, config, budget, delta, epsilon, method="iqae"):
    theta = np.array([0., np.pi/2])
    remaining = budget
    inconsistent = False
    for _ in range(config.max_rounds):
        k, branch = safe_power(theta, config.max_grover_power)
        k = min(k, (remaining-1)//2)
        if k < 0:
            break
        k, branch = safe_power(theta, k)
        shots = min(config.shots_per_round, remaining//(2*k+1))
        if shots == 0:
            break
        h = measurement.sample(k, shots)
        remaining -= shots*(2*k+1)
        interval = invert_interval(clopper_pearson(h, shots, delta/config.max_rounds), k, branch)
        new = np.array([max(theta[0], interval[0]), min(theta[1], interval[1])])
        if new[0] > new[1]:
            inconsistent = True
            break
        theta = new
        ci = np.sin(theta)**2
        measurement.rounds[-1].update(theta_interval=theta.tolist(), amplitude_interval=ci.tolist(), branch=branch)
        if config.stopping_rule == "interval" and (ci[1]-ci[0])/2 <= epsilon:
            break
    ci = tuple(map(float, np.sin(theta)**2))
    measurement.finalize_resources()
    return AmplitudeResult(float(np.mean(ci)), ci, "cp_alpha_spending" if not inconsistent else "inconsistent_intervals",
                           not inconsistent and (ci[1]-ci[0])/2 <= epsilon, measurement.ledger, measurement.rounds)

def official_reference(oracle, epsilon=.01, delta=.05, seed=13, shots=200):
    """Qiskit reference implementation, intentionally outside bounded experiments."""
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
    problem = EstimationProblem(oracle.circuit, objective_qubits=[oracle.n])
    estimator = IterativeAmplitudeEstimation(epsilon, delta, sampler=StatevectorSampler(default_shots=shots, seed=seed))
    result = estimator.estimate(problem)
    return {"estimate": result.estimation, "interval": list(result.confidence_interval), "oracle_queries_qiskit": result.num_oracle_queries}
