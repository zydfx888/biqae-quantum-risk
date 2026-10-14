import numpy as np
from q3_biqae.suite_v4_binning import compress, exact_metrics
from q3_biqae.scenario_table import ScenarioTable
from q3_biqae.exact_reference import risk, gradient

def test_mass_and_all_asset_means_preserved():
    x=np.random.default_rng(11).normal(size=(1003,4));w=np.array([.4,0,0,.6])
    for method in ['uniform_mass_actual','uniform_width_actual','tail95_actual','tail90_offgrid_actual']:
        t=compress(x,w,32,method)
        np.testing.assert_allclose(t.probabilities.sum(),1,atol=1e-14)
        np.testing.assert_allclose(t.probabilities@t.returns,x.mean(axis=0),atol=1e-14)

def test_aligned_tail_structurally_preserves_anchor_es_and_gradient_but_not_other_weights():
    x=np.random.default_rng(18).normal(size=(2003,4));w=np.array([.4,0,0,.6])
    raw=ScenarioTable(x,np.full(len(x),1/len(x)));small=compress(x,w,16,'tail95_actual')
    for alpha in [.95,.975]:
        np.testing.assert_allclose(risk(small.losses(w),small.probabilities,alpha)['es'],risk(raw.losses(w),raw.probabilities,alpha)['es'],atol=1e-12)
        np.testing.assert_allclose(gradient(small,w,alpha),gradient(raw,w,alpha),atol=1e-12)
    other=np.array([0,.6,.4,0])
    assert abs(exact_metrics(raw,small,other,.975)['es_bias_bp'])>100
