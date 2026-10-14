from dataclasses import dataclass, asdict

@dataclass
class ResourceLedger:
    samples: int = 0
    shots_total: int = 0
    grover_applications: int = 0
    oracle_A_calls: int = 0
    A_forward_calls: int = 0
    A_inverse_calls: int = 0
    max_grover_power: int = 0
    max_depth: int = 0
    max_two_qubit_gates: int = 0
    total_two_qubit_gates: int = 0
    qubits: int = 0
    runtime_ms: float = 0.
    compilation_ms: float = 0.

    def record(self, shots, k):
        self.shots_total += shots
        self.grover_applications += shots*k
        self.A_forward_calls += shots*(k+1)
        self.A_inverse_calls += shots*k
        self.oracle_A_calls += shots*(2*k+1)
        self.max_grover_power = max(self.max_grover_power, k)

    def merge(self, other):
        for name, value in asdict(other).items():
            if name.startswith("max_") or name == "qubits":
                setattr(self, name, max(getattr(self, name), value))
            else:
                setattr(self, name, getattr(self, name)+value)
        return self

    def to_dict(self):
        return asdict(self)

