import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from q3_biqae.state_preparation import probability_circuit
from q3_biqae.tail_oracle import TailOracle

@pytest.mark.parametrize("states", [8, 16, 32])
def test_loading_and_inverse(states):
    p = np.arange(1, states+1, dtype=float)
    p /= p.sum()
    circuit = probability_circuit(p)
    np.testing.assert_allclose(Statevector.from_instruction(circuit).probabilities(), p, atol=1e-10)
    oracle = TailOracle(p, np.linspace(0, 1, states))
    assert oracle.validate()["inverse_error"] < 1e-8

def test_grover_likelihood():
    oracle = TailOracle([.2, .8], [0., 1.])
    theta = np.arcsin(np.sqrt(.8))
    for k in (0, 1, 2, 3):
        actual = Statevector.from_instruction(oracle.amplified(k)).probabilities([oracle.n])[1]
        assert actual == pytest.approx(np.sin((2*k+1)*theta)**2, abs=1e-8)

