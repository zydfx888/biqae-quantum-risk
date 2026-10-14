import numpy as np
from q3_biqae.trained_model import train_mixture, TrainedRiskModel, freeze_model_scenarios

def test_training_likelihood_and_reload(tmp_path):
    rng = np.random.default_rng(5)
    x = np.r_[rng.normal(0, .01, (400, 4)), rng.normal(-.015, .03, (100, 4))]
    model, history = train_mixture(x, iterations=100)
    scores = [h["training_average_log_likelihood_standardized"] for h in history]
    assert scores[-1] > scores[0]
    assert np.min(np.diff(scores)) > -1e-5
    path = tmp_path/"model.npz"
    model.save(path)
    reloaded = TrainedRiskModel.load(path)
    np.testing.assert_array_equal(model.sample(100, 7), reloaded.sample(100, 7))
    np.testing.assert_array_equal(model.log_density(x), reloaded.log_density(x))

def test_generated_scenarios_frozen_without_mass_loss():
    x = np.random.default_rng(8).normal(0, .01, (20003, 4))
    table = freeze_model_scenarios(x)
    assert len(table.probabilities) == 16
    np.testing.assert_allclose(table.probabilities@table.returns, x.mean(axis=0), atol=1e-12)
