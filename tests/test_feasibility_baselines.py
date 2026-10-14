import numpy as np
import pytest
from q3_biqae.feasibility_baselines import estimate_classical_risk
from q3_biqae.exact_reference import risk
from q3_biqae.scenario_table import ScenarioTable

@pytest.mark.parametrize("method", ["mc", "sobol", "is"])
def test_single_sample_risk_baseline(table, method):
    weights = np.full(4, .25)
    result = estimate_classical_risk(table, weights, .85, 65536, 13, method)
    truth = risk(table.losses(weights), table.probabilities, .85)
    assert abs(result['es']-truth['es']) < .15
    assert result['samples'] == 65536
    assert result['ci']['es'][0] <= truth['es'] <= result['ci']['es'][1]
    assert result['ci']['var'][0] <= truth['var'] <= result['ci']['var'][1]

def test_unsampled_extreme_state_remains_in_var_confidence_bound():
    table = ScenarioTable([[0, 0], [-1, -1]], [.97, .03])
    result = estimate_classical_risk(table, [.5, .5], .975, 16, 0, 'mc')
    assert result['var'] == 0
    assert result['ci']['var'] == [0., 1.]
