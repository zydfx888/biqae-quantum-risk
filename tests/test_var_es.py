import numpy as np
from q3_biqae.config import Config
from q3_biqae.es_estimation import estimate_risk
from q3_biqae.exact_reference import risk
from q3_biqae.var_search import search_var

def test_unresolved_quantile_reports_range(table):
    cfg = Config(budget=32, max_rounds=2)
    losses = table.losses([.25]*4)
    result = search_var(losses, table.probabilities, .95, "iqae", cfg, 12, 13)
    assert not result["resolved"]
    assert result["interval"][0] <= 10 <= result["interval"][1]
    assert result["ledger"].oracle_A_calls <= 12

def test_end_to_end_discrete_risk(table):
    cfg = Config(max_rounds=30, shots_per_round=200, epsilon=.005)
    truth = risk(table.losses([.25]*4), table.probabilities, .85)
    result = estimate_risk(table, .85, 200000, 13, "iqae", cfg)
    assert result["ci"]["var"][0] <= truth["var"] <= result["ci"]["var"][1]
    assert result["ci"]["es"][0] <= truth["es"] <= result["ci"]["es"][1]
    assert result["resources"]["oracle_A_calls"] <= 200000

