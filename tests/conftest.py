import numpy as np
import pytest
from q3_biqae.scenario_table import ScenarioTable
from q3_biqae.config import ROOT

def pytest_configure(config):
    # Keep tests independent of global pytest temporary directories as well.
    import uuid
    if config.option.basetemp is None:
        config.option.basetemp = str(ROOT/"tmp"/("pytest_"+uuid.uuid4().hex))

@pytest.fixture
def table():
    losses = np.array([-2., -1., 0., 1., 1., 3., 5., 10.])
    return ScenarioTable(-np.tile(losses[:, None], (1, 4)), np.array([.1, .1, .2, .2, .1, .1, .1, .1]))
