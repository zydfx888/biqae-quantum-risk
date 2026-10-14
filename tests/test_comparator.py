import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from q3_biqae.tail_oracle import comparator
from q3_biqae.loss_oracle import loss_lookup

@pytest.mark.parametrize("threshold", [-1, 0, 1, 3, 7, 8])
def test_comparator_all_basis_states(threshold):
    circuit = comparator(3, threshold)
    for loss in range(8):
        state = Statevector.from_int(loss, 16).evolve(circuit)
        expected = loss + (8 if loss >= threshold else 0)
        assert np.abs(state.data[expected])**2 == pytest.approx(1)

def test_loss_lookup_and_uncompute():
    levels = [0, 3, 3, 7]
    circuit = loss_lookup(levels)
    for j, level in enumerate(levels):
        initial = Statevector.from_int(j, 2**circuit.num_qubits)
        computed = initial.evolve(circuit)
        assert abs(computed.data[j+(level << 2)])**2 == pytest.approx(1)
        np.testing.assert_allclose(computed.evolve(circuit.inverse()).data, initial.data, atol=1e-10)

