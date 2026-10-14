import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

def probability_circuit(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) < 2 or np.any(p < 0) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Invalid probability table")
    n = int(np.ceil(np.log2(len(p))))
    amplitudes = np.zeros(2**n)
    amplitudes[:len(p)] = np.sqrt(p)
    circuit = QuantumCircuit(n, name="P")
    circuit.append(StatePreparation(amplitudes), range(n))
    return circuit

