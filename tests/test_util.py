import numpy as np
from numpy.testing import assert_array_equal

from pareto.util import all_pareto_fronts, parse_ranges


def test_all_pareto_fronts():
    x = np.array([[[1.1, 0.0], [0.0, 0.0], [1.0, 1.0]]])
    y = all_pareto_fronts(x)
    expected = np.array([[True, False, True]])
    assert_array_equal(y, expected)


def test_parse_ranges():
    j = """
    {
        "C": "Real(1e-6, 1e+6, prior='log-uniform')",
        "gamma": "Real(high=1e+1, low=1e-6, prior='log-uniform')",
        "degree": "Integer(1,8)",
        "test": "('Hallo', 'Welt')"
    }
    """
    result = parse_ranges(j)

    assert len(result) == 4
