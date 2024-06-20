import inspect
from tensorflow.keras.optimizers import *


__all__ = [
    "AdadeltaSK",
    "AdagradSK",
    "AdamSK",
    "AdamaxSK",
    "FtrlSK",
    "NadamSK",
    "RMSpropSK",
    "SGDSK",
]


def wrap_class(cl):
    class Wrapper(cl):
        def __init__(self, **parameters):
            p = list(inspect.signature(super().__init__).parameters.keys())
            p.remove("name")
            p.remove("kwargs")
            if "initial_accumulator_value" in p:
                p.remove("initial_accumulator_value")
            if "l2_shrinkage_regularization_strength" in p:
                p.remove("l2_shrinkage_regularization_strength")
            self.sk_params = p
            super().__init__(**parameters)

        def get_params(self, **kwargs):
            d = {}
            for p in self.sk_params:
                d[p] = getattr(self, p)
            return d

        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                if parameter in self.sk_params:
                    setattr(self, parameter, value)
            return self

    return Wrapper


AdadeltaSK = wrap_class(Adadelta)
AdagradSK = wrap_class(Adagrad)
AdamSK = wrap_class(Adam)
AdamaxSK = wrap_class(Adamax)
FtrlSK = wrap_class(Ftrl)
NadamSK = wrap_class(Nadam)
RMSpropSK = wrap_class(RMSprop)
SGDSK = wrap_class(SGD)
