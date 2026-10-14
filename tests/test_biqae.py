import numpy as np
import pytest
from q3_biqae.config import Config
from q3_biqae.amplitude import estimate_payload

@pytest.mark.parametrize("method", ["grid_biqae", "normal_biqae", "beta_biqae"])
@pytest.mark.parametrize("a", [.01, .025, .05, .4, .99])
def test_bayesian_posterior_and_reproducibility(method, a):
    cfg = Config(max_rounds=25, shots_per_round=200)
    result = estimate_payload(np.array([1-a, a]), np.array([0., 1.]), method, cfg, 30000, 29)
    assert abs(result.estimate-a) < .03
    assert result.ledger.oracle_A_calls <= 30000
    for row in result.rounds:
        assert sum(row["posterior"]) == pytest.approx(1.)
        assert 0 <= row["amplitude_interval"][0] <= row["amplitude_interval"][1] <= 1
    repeat = estimate_payload(np.array([1-a, a]), np.array([0., 1.]), method, cfg, 30000, 29)
    assert repeat.estimate == result.estimate

