import numpy as np
import pytest
from q3_biqae.amplitude import estimate_payload
from q3_biqae.config import Config
from q3_biqae.statistics import holm

@pytest.mark.parametrize("method", ["mc", "sobol", "is", "mlae"])
def test_classical_and_mlae_amplitudes(method):
    result = estimate_payload(np.array([.7, .3]), np.array([0., 1.]), method, Config(), 10000, 13)
    assert abs(result.estimate-.3) < .05
    assert result.ledger.samples+result.ledger.oracle_A_calls <= 10000

def test_holm():
    assert holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])

def test_sobol_zero_replicate_variance_has_discrepancy_guard():
    result = estimate_payload(np.array([.50000001, .49999999]), np.array([0., 1.]), "sobol", Config(), 128, 13)
    assert result.interval[0] < .49999999 < result.interval[1]

@pytest.mark.parametrize("method", ["mc", "sobol", "is", "mlae", "iqae", "grid_biqae", "normal_biqae", "beta_biqae"])
def test_last_single_query_budget(method):
    result = estimate_payload(np.array([.7, .3]), np.array([0., 1.]), method, Config(), 1, 13)
    assert 0 <= result.estimate <= 1
    assert result.ledger.samples+result.ledger.oracle_A_calls == 1
    assert result.interval[0] <= result.interval[1]
