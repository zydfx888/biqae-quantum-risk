import numpy as np
import pytest
from q3_biqae.tail_oracle import TailOracle, excess_payload
from q3_biqae.cvar_gradient import joint_gradient
from q3_biqae.config import Config
from q3_biqae.exact_reference import cvar

def test_excess_rotations():
    p = np.array([.2, .3, .5])
    values = excess_payload(np.array([-2., 1., 4.]), 1., 6.)
    assert TailOracle(p, values).validate()["target_probability"] == pytest.approx(.25)
    with pytest.raises(ValueError):
        excess_payload(np.array([5.]), 1., 3.)

def test_joint_gradient_finite_difference(table):
    w, tau = np.full(4, .25), 2.
    g, gt, _, _ = joint_gradient(table, w, tau, .85, "exact", Config(), 100, 1)
    h = 1e-6
    for i in range(4):
        step = np.eye(4)[i]*h
        derivative = (cvar(table, w+step, .85, tau)-cvar(table, w-step, .85, tau))/(2*h)
        assert g[i] == pytest.approx(derivative, rel=1e-6)
    derivative = (cvar(table, w, .85, tau+h)-cvar(table, w, .85, tau-h))/(2*h)
    assert gt == pytest.approx(derivative, rel=1e-6)

