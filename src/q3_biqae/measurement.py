from time import perf_counter
import numpy as np
from .resource_ledger import ResourceLedger

class Measurement:
    def __init__(self, probabilities, payload, config, seed):
        self.p, self.y = np.asarray(probabilities), np.asarray(payload)
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.ledger = ResourceLedger()
        self.rounds = []
        self.oracle = None
        self.compiled = {}
        self._ideal_probabilities = {}
        if config.backend != "analytic_shots":
            from .tail_oracle import TailOracle
            self.oracle = TailOracle(self.p, self.y)
            self.oracle.validate()

    def sample(self, k, shots):
        started = perf_counter()
        raw_counts = None
        if self.config.backend == "analytic_shots":
            # Explicit classical simulation of the ideal Bernoulli likelihood;
            # this amplitude is never exposed to the estimator or scheduler.
            theta = np.arcsin(np.sqrt(np.clip(self.p @ self.y, 0, 1)))
            probability = np.sin((2*k+1)*theta)**2
            h = int(self.rng.binomial(shots, probability))
        elif self.config.backend == "qiskit_statevector":
            from qiskit.quantum_info import Statevector
            if k not in self._ideal_probabilities:
                self._ideal_probabilities[k] = Statevector.from_instruction(self.oracle.amplified(k)).probabilities([self.oracle.n])[1]
            h = int(self.rng.binomial(shots, self._ideal_probabilities[k]))
        else:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            from qiskit import transpile
            model = None
            if self.config.backend == "aer_noise":
                model = NoiseModel()
                model.add_all_qubit_quantum_error(depolarizing_error(self.config.noise_probability, 1), ["u"])
                model.add_all_qubit_quantum_error(depolarizing_error(self.config.noise_probability, 2), ["cx"])
            simulator = AerSimulator(noise_model=model)
            if k not in self.compiled:
                tick = perf_counter()
                self.compiled[k] = transpile(self.oracle.amplified(k, True), basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=0)
                self.ledger.compilation_ms += (perf_counter()-tick)*1000
            raw_counts = simulator.run(self.compiled[k], shots=shots, seed_simulator=int(self.rng.integers(2**31))).result().get_counts()
            h = int(raw_counts.get("1", 0))
        self.ledger.record(shots, k)
        self.ledger.runtime_ms += (perf_counter()-started)*1000
        self.rounds.append({"k": k, "shots": shots, "successes": h,
                            "oracle_A_calls_cumulative": self.ledger.oracle_A_calls,
                            "raw_counts": raw_counts})
        return h

    def finalize_resources(self):
        if self.oracle is None:
            return  # Analytic simulation must not invent circuit measurements.
        from qiskit import transpile
        for k in {r["k"] for r in self.rounds}:
            if k not in self.compiled:
                tick = perf_counter()
                self.compiled[k] = transpile(self.oracle.amplified(k), basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=0)
                self.ledger.compilation_ms += (perf_counter()-tick)*1000
            qc = self.compiled[k]
            gates = int(qc.count_ops().get("cx", 0))
            self.ledger.max_depth = max(self.ledger.max_depth, qc.depth())
            self.ledger.max_two_qubit_gates = max(self.ledger.max_two_qubit_gates, gates)
            self.ledger.total_two_qubit_gates += sum(r["shots"] for r in self.rounds if r["k"] == k)*gates
            self.ledger.qubits = qc.num_qubits

