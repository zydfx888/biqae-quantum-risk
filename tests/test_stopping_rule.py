import pytest
from dataclasses import replace
from q3_biqae.config import Config
from q3_biqae.amplitude import estimate_payload

@pytest.mark.parametrize('method', ['iqae', 'grid_biqae', 'normal_biqae', 'beta_biqae'])
def test_budget_stop_spends_but_default_unchanged(method):
    cfg = Config(grid_size=2049, max_grover_power=31, shots_per_round=64,
                 max_rounds=256, record_posterior=False)
    args = ([.7, .3], [0., 1.], method)
    default = estimate_payload(*args, cfg, 2048, 13, epsilon=.49)
    explicit = estimate_payload(*args, replace(cfg, stopping_rule='interval'), 2048, 13, epsilon=.49)
    fixed = estimate_payload(*args, replace(cfg, stopping_rule='budget'), 2048, 13, epsilon=.49)
    assert default.estimate == explicit.estimate
    assert default.interval == explicit.interval
    assert default.ledger.oracle_A_calls == explicit.ledger.oracle_A_calls
    assert fixed.ledger.oracle_A_calls == 2048
    assert fixed.ledger.oracle_A_calls > default.ledger.oracle_A_calls

def test_unknown_stopping_rule():
    with pytest.raises(ValueError):
        Config(stopping_rule='unknown')
