import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import raises

from pareto.data import DTLZ, ZDT, TwoParabola


def check_X_y(X, y, expected_X_shape, expected_y_shape):
    assert X.shape == expected_X_shape
    assert y.shape == expected_y_shape
    assert isinstance(X[0][0][0], np.float32)
    assert isinstance(y[0][0], np.bool_)


def test_two_parabola():
    # Test if it works with defaults:
    tp = TwoParabola(n_instances=1, n_objects=2)
    X, y = tp.get_xy()
    check_X_y(X, y, (1, 2, 2), (1, 2))

    # The problem is only defined for 2 features:
    tp = TwoParabola(n_instances=1, n_objects=1, n_features=1)
    with raises(ValueError):
        tp.get_xy()


@pytest.mark.parametrize("prob_id", list(range(1, 8)))
def test_dtlz(prob_id):
    dtlz = DTLZ(
        n_features=4, n_objectives=3, n_instances=1, n_objects=2, prob_id=prob_id
    )
    X, y = dtlz.get_xy()
    check_X_y(X, y, (1, 2, 4), (1, 2))


def test_dtlz_raises():
    with raises(ValueError):
        DTLZ(prob_id=8)
    with raises(ValueError):
        DTLZ(n_objectives=1)
    with raises(ValueError):
        DTLZ(n_objectives=2, n_features=2)


@pytest.mark.parametrize("prob_id", (1, 2, 3, 4, 6))
def test_zdt(prob_id):
    zdt = ZDT(n_features=4, n_instances=1, n_objects=2, prob_id=prob_id)
    X, y = zdt.get_xy()
    check_X_y(X, y, (1, 2, 4), (1, 2))


def test_zdt_prob5():
    zdt = ZDT(n_features=35, n_instances=1, n_objects=2, prob_id=5)
    X, y = zdt.get_xy()
    check_X_y(X, y, (1, 2, 35), (1, 2))


def test_zdt_raises():
    with raises(ValueError):
        ZDT(prob_id=7)
    with raises(ValueError):
        ZDT(n_features=1, prob_id=5)
