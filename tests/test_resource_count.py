import numpy as np
import pytest
from q3_biqae.resource_ledger import ResourceLedger
from q3_biqae.config import Config, guarded
from q3_biqae.optimizer import project_weights

def test_expanded_queries():
    ledger = ResourceLedger()
    ledger.record(100, 0)
    ledger.record(50, 3)
    assert ledger.shots_total == 150
    assert ledger.grover_applications == 150
    assert ledger.oracle_A_calls == 450
    assert ledger.A_forward_calls == 300
    assert ledger.A_inverse_calls == 150

def test_boundary_guard():
    with pytest.raises(ValueError):
        guarded("../outside.json")
    with pytest.raises(ValueError):
        Config(output="../outside")

def test_projection():
    for values in ([9., -4., 1., 2.], [0., 0., 0., 0.]):
        projected = project_weights(values, .4)
        assert projected.sum() == pytest.approx(1)
        assert np.all((projected >= 0) & (projected <= .4))

