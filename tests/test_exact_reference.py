import numpy as np
import pytest
from q3_biqae.exact_reference import risk, gradient, cvar
from q3_biqae.scenario_table import ScenarioTable
from q3_biqae.classical_cvar import solve_cvar

def test_atom_es_not_conditional_mean():
    result = risk(np.array([0., 1., 5.]), np.array([.8, .15, .05]), .9)
    assert result["var"] == 1
    assert result["es"] == pytest.approx(3)

def test_boundary_and_duplicate_quantile(table):
    losses = table.losses(np.full(4, .25))
    assert risk(losses, table.probabilities, .2)["var"] == -1
    assert risk(losses, table.probabilities, .6)["var"] == 1

def test_reject_invalid_table():
    with pytest.raises(ValueError):
        ScenarioTable([[0], [1]], [.2, .7])
    with pytest.raises(ValueError):
        ScenarioTable([[0], [float('nan')]], [.5, .5])

def test_lp_and_gradient(table):
    weights = np.full(4, .25)
    value = cvar(table, weights, .85)
    result = solve_cvar(table, .85)
    assert result["cvar"] == pytest.approx(value)
    assert gradient(table, weights, .85) @ weights == pytest.approx(value)

