"""Exact table loading and fused controlled rotations. No QRAM assumption."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from .state_preparation import probability_circuit

def comparator(bits, threshold):
    qc = QuantumCircuit(bits+1, name="ge_threshold")
    for value in range(2**bits):
        if value < threshold:
            continue
        zeros = [i for i in range(bits) if not (value >> i)&1]
        for i in zeros:
            qc.x(i)
        qc.mcx(list(range(bits)), bits)
        for i in zeros:
            qc.x(i)
    return qc

def excess_payload(losses, tau, bound):
    excess = np.maximum(np.asarray(losses)-tau, 0)
    if bound <= 0:
        if np.any(excess):
            raise ValueError("Nonzero excess with zero bound")
        return np.zeros_like(excess)
    if np.max(excess) > bound+1e-12:
        raise ValueError("Frozen loss bound exceeded")
    return excess/bound

class TailOracle:
    def __init__(self, probabilities, payload):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.payload = np.asarray(payload, dtype=float)
        if self.payload.shape != self.probabilities.shape or not np.isfinite(self.payload).all() or np.any((self.payload < 0)|(self.payload > 1)):
            raise ValueError("Payload must be in [0,1]")
        self.preparation = probability_circuit(self.probabilities)
        self.n = self.preparation.num_qubits
        self.circuit = QuantumCircuit(self.n+1, name="A")
        self.circuit.compose(self.preparation, range(self.n), inplace=True)
        for j, value in enumerate(self.payload):
            if value == 0:
                continue
            zeros = [i for i in range(self.n) if not (j >> i)&1]
            for i in zeros:
                self.circuit.x(i)
            self.circuit.append(RYGate(2*np.arcsin(np.sqrt(value))).control(self.n, annotated=True), range(self.n+1))
            for i in zeros:
                self.circuit.x(i)

    def amplified(self, k, measure=False):
        from qiskit.circuit.library import grover_operator
        qc = self.circuit.copy()
        if k:
            marker = QuantumCircuit(self.n+1)
            marker.z(self.n)
            grover = grover_operator(marker, state_preparation=self.circuit)
            for _ in range(k):
                qc.compose(grover, inplace=True)
        if measure:
            qc.add_register(__import__("qiskit").ClassicalRegister(1))
            qc.measure(self.n, 0)
        return qc

    def validate(self):
        from qiskit.quantum_info import Statevector
        state = Statevector.from_instruction(self.circuit)
        actual = float(state.probabilities([self.n])[1])
        expected = float(self.probabilities @ self.payload)
        restored = state.evolve(self.circuit.inverse()).data
        target = np.zeros_like(restored)
        target[0] = 1
        error = float(np.max(np.abs(restored-target)))
        if abs(actual-expected) > 1e-8 or error > 1e-8:
            raise RuntimeError("Oracle failed statevector acceptance")
        return {"target_probability": actual, "probability_error": abs(actual-expected), "inverse_error": error}
