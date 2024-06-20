import pytest
from csrank import FATEChoiceFunction
from tensorflow.keras.optimizers import Adam
from pareto.experiments.wrappers import *

def test_optimizer_in_estimator():
    fate = FATEChoiceFunction(optimizer=Adam)
    params = fate.get_params()
    print(params)


def test_wrapper_in_estimator():
    fate = FATEChoiceFunction(optimizer=AdamSK)
    params = fate.get_params()


@pytest.mark.parametrize(
    "optimizer,parameters",
    [
        (AdadeltaSK, ["learning_rate", "rho", "epsilon"]),
        (AdagradSK, ["learning_rate", "epsilon"]),
        (AdamSK, ["learning_rate", "beta_1", "beta_2", "epsilon", "amsgrad"]),
        (AdamaxSK, ["learning_rate", "beta_1", "beta_2", "epsilon"]),
        (
            FtrlSK,
            [
                "learning_rate",
                "learning_rate_power",
                "l1_regularization_strength",
                "l2_regularization_strength",
            ],
        ),
        (NadamSK, ["learning_rate", "beta_1", "beta_2", "epsilon"]),
        (RMSpropSK, ["learning_rate", "rho", "momentum", "epsilon", "centered"]),
        (SGDSK, ["learning_rate", "momentum", "nesterov"]),
    ],
)
def test_optimizer(optimizer, parameters):
    opt = optimizer()
    for p in parameters:
        get_p = opt.get_params()
        assert hasattr(opt, p)
        assert p in opt.sk_params
        assert p in get_p
