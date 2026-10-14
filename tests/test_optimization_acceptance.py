import numpy as np
import pytest
from q3_biqae.scenario_table import ScenarioTable
from q3_biqae.classical_cvar import solve_cvar
from q3_biqae.optimization_acceptance import solve_sparse

@pytest.mark.parametrize('seed',[7,19])
def test_sparse_lp_matches_independent_dense_reference(seed):
    rng=np.random.default_rng(seed)
    table=ScenarioTable(rng.normal(0,.02,(25,4)),rng.dirichlet(np.ones(25)))
    sparse=solve_sparse(table);dense=solve_cvar(table,.975,.6)
    assert sparse['objective']==pytest.approx(dense['cvar'],abs=1e-10)
    w=np.array(sparse['weights'])
    assert abs(w.sum()-1)<1e-10 and w.min()>=-1e-10 and w.max()<=.6+1e-10
