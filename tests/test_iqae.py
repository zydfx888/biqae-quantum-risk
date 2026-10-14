import numpy as np
import pytest
from q3_biqae.config import Config
from q3_biqae.amplitude import estimate_payload
from q3_biqae.iqae import safe_power, invert_interval

@pytest.mark.parametrize("a", [0., .01, .025, .05, .3, .8, .99, 1.])
def test_iqae_known_amplitudes(a):
    config = Config(max_rounds=30, shots_per_round=200, max_grover_power=31)
    result = estimate_payload(np.array([1-a, a]), np.array([0., 1.]), "iqae", config, 100000, 13)
    assert result.interval[0]-1e-12 <= a <= result.interval[1]+1e-12
    assert result.converged
    assert result.ledger.oracle_A_calls <= 100000

def test_monotone_branch_inversion():
    theta = np.array([.51, .52])
    k, branch = safe_power(theta, 31)
    p = np.sort(np.sin((2*k+1)*theta)**2)
    np.testing.assert_allclose(invert_interval(p, k, branch), theta, atol=1e-12)

