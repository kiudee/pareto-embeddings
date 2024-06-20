import pytest
import torch

from pareto.models import FATEParetoEmbedder
from pareto.util import all_pareto_fronts


@pytest.fixture
def minimal_problem():
    torch.manual_seed(0)
    # (100 instances, 5 objects, 4 features)
    X = torch.rand(100, 5, 4, dtype=torch.float32)
    scores = X.mean(2)[..., None]
    Y = torch.tensor(all_pareto_fronts(scores.numpy()))
    return X, Y


def test_fate_pareto_embedder(minimal_problem):
    X, Y = minimal_problem
    n_inst, n_obj, n_feat = X.shape
    fpe = FATEParetoEmbedder(
        module__n_input_features=n_feat,
        module__set_encoder_args={"dim": 8},
        callbacks=[],
        lr=0.01,
        max_epochs=10,
        batch_size=10,
        train_split=None,
    )
    fpe.fit(X, Y)
