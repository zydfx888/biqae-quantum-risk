import pytest
from q3_biqae.suite_risk import cp, decompose_es

def test_reliability_distinguishes_observed_rate_from_confidence():
    assert cp(95,100)[0] < .9
    assert cp(99,100)[0] < .95
    assert cp(100,100)[0] > .95
    assert cp(100,100,.05/15)[0] < .95

def test_signed_error_decomposition_including_negative_clip_correction():
    result = decompose_es(.023,.025,.024,.027)
    assert result['discretization_bp']==pytest.approx(-30)
    assert result['threshold_bp']==pytest.approx(10)
    assert result['sampling_amplitude_and_clip_bp']==pytest.approx(-20)
    assert result['total_vs_continuous_bp']==pytest.approx(-40)
