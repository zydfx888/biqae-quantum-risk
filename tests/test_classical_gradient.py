from dataclasses import replace
import numpy as np
import pytest
from q3_biqae.classical_gradient import shared_batch_gradient
from q3_biqae.scenario_table import ScenarioTable
from q3_biqae.config import Config
from q3_biqae.cvar_gradient import joint_gradient
from q3_biqae.feasibility import policy_train


@pytest.fixture
def correlated_table():
    z = np.array([-.04, -.01, .02, .05])
    return ScenarioTable(np.column_stack([z, -2*z, 3*z, np.zeros(4)]), np.full(4, .25))


@pytest.mark.parametrize("method", ["mc", "sobol", "is"])
def test_shared_batch_preserves_signed_vector_relations(correlated_table, method):
    g, _, ledger, logs = shared_batch_gradient(correlated_table, [.25, 1, 0, 0], .9, 1024, 13, method)
    # One observed vector must preserve exact cross-asset identities. Independent
    # sampling of signed components would break these identities stochastically.
    assert g[1] == pytest.approx(-2*g[0], abs=1e-14)
    assert g[2] == pytest.approx(3*g[0], abs=1e-14)
    assert g[3] == 0
    assert ledger.samples == 1024
    assert ledger.oracle_A_calls == ledger.shots_total == 0
    assert len(logs) == len(logs[0]["rounds"]) == 1
    assert sum(logs[0]["rounds"][0]["state_counts"]) == 1024


def test_sobol_exact_dyadic_mass_including_fractional_atom(correlated_table):
    tail = np.array([.25, 1, 0, 0])
    g, gt, _, _ = shared_batch_gradient(correlated_table, tail, .9, 16, 13, "sobol")
    expected = -(correlated_table.probabilities*tail)@correlated_table.returns/.1
    np.testing.assert_allclose(g, expected, atol=1e-14)
    assert gt == pytest.approx(1-correlated_table.probabilities@tail/.1)


@pytest.mark.parametrize("method", ["mc", "sobol", "is"])
def test_single_vector_budget_serves_weights_and_tau(correlated_table, method):
    _, _, ledger, logs = joint_gradient(correlated_table, [.25]*4, 0, .9, method, Config(), 1, 13)
    assert ledger.samples == 1
    assert logs[0]["component"] == "all_weights_and_tau"


def test_is_uses_single_proposal_and_ordinary_likelihood_ratio(correlated_table):
    tail = np.array([.25, 1, 0, 0])
    g, gt, _, logs = shared_batch_gradient(correlated_table, tail, .9, 1024, 29, "is")
    details = logs[0]["rounds"][0]
    q = np.array(details["proposal_probabilities"])
    mass = np.array(details["state_counts"])*correlated_table.probabilities/q/1024
    np.testing.assert_allclose(g, -(mass*tail)@correlated_table.returns/.1, atol=1e-14)
    assert gt == pytest.approx(1-mass@tail/.1)
    assert np.all(q >= .5*correlated_table.probabilities)


@pytest.mark.parametrize("method", ["mc", "sobol", "is"])
def test_training_bypasses_component_amplitude_sampling(monkeypatch, correlated_table, method):
    def forbidden(*args, **kwargs):
        raise AssertionError("Classical gradients must not use per-component amplitude calls")
    monkeypatch.setattr("q3_biqae.feasibility.estimate_payload", forbidden)
    cfg = Config(budget=128, optimization_steps=3)
    result = policy_train(correlated_table, cfg, method, 13, .1)
    assert result["training_resources"]["samples"] == 3*128
    assert result["gradient_sampling"] == "shared_vector_batch"
    assert sum(result["weights"]) == pytest.approx(1)


def test_nondyadic_qmc_budget_and_invalid_tail(correlated_table):
    result = shared_batch_gradient(correlated_table, [1, 0, 0, 0], .9, 100, 13, "sobol")
    assert result[2].samples == 64
    with pytest.raises(ValueError):
        shared_batch_gradient(correlated_table, [1, 2, 0, 0], .9, 100, 13, "mc")
