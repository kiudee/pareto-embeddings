import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from pareto.layers import FATE, FSEncoder, MeanEncoder


@pytest.fixture
def rand():
    return np.random.default_rng(0)


@pytest.mark.parametrize("set_encoder", (FSEncoder, MeanEncoder))
def test_fate_permutation(rand, set_encoder):
    n_features = 1
    X = torch.tensor(rand.normal(size=(1, 2, n_features)).astype(np.float32))
    net = FATE(
        set_encoder=set_encoder,
        n_input_features=n_features,
        n_scorer_dim=1,
        n_scorer_layers=1,
        n_scorer_output=1,
        set_encoder_args=dict(dim=1),
    )
    result1 = net(X).detach().numpy()
    X_permuted = X[:, [1, 0], :]
    result2 = net(X_permuted).detach().numpy()
    assert_array_equal(result1, result2[:, [1, 0], :])
