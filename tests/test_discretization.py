import numpy as np
from q3_biqae.distributions import generate
from q3_biqae.discretization import discretize

def test_mass_mean_and_reproducibility():
    raw = generate("synthetic_student_t", 1000, 4, 13)
    copy = generate("synthetic_student_t", 1000, 4, 13)
    assert raw.sha256 == copy.sha256
    reduced, _ = discretize(raw, [.25]*4, 16, .975)
    assert len(reduced.probabilities) == 16
    assert np.isclose(reduced.probabilities.sum(), 1)
    np.testing.assert_allclose(raw.probabilities @ raw.returns, reduced.probabilities @ reduced.returns, atol=1e-12)
    assert np.isclose(reduced.probabilities[-8:].sum(), .05)

def test_generators():
    for kind in ("normal", "student_t", "t_copula", "regime", "extreme"):
        assert generate(kind, 100, 4, 1).returns.shape == (100, 4)

