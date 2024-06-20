import torch
from skorch import NeuralNet
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import OneCycleLR

from pareto.layers import FATE, FSEncoder, MeanEncoder, ParetoEmbedding, PairwiseEncoder
from pareto.losses import ParetoEmbeddingLoss, ParetoEmbeddingLossSoft
from pareto.metrics_np import instance_informedness_np
from pareto.util import all_pareto_fronts

__all__ = [
    "ParetoEmbedder",
    "FATEParetoEmbedder",
    "ParetoEmbedderSGD",
    "ParetoEmbedderSGDSoft",
    "PWParetoEmbedder",
    "PWParetoEmbedderSoft",
]

ENCODERS = {"fspool": FSEncoder, "mean": MeanEncoder}


class ParetoEmbedder(NeuralNet):
    def __init__(
        self,
        *args,
        module=ParetoEmbedding,
        criterion=ParetoEmbeddingLoss,
        optimizer=None,
        callbacks=None,
        **kwargs
    ):
        if optimizer is None:
            optimizer = torch.optim.AdamW
        if callbacks is None:
            callbacks = [
                (
                    "lr_scheduler",
                    LRScheduler(
                        policy=OneCycleLR, max_lr=0.01, epochs=1000, steps_per_epoch=10
                    ),
                )
            ]
        super().__init__(
            *args,
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            **kwargs
        )

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self.criterion_(y_pred, y_true, X=X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return instance_informedness_np(y_pred, y)

    def predict(self, X):
        scores = super().predict(X)
        return all_pareto_fronts(scores)


class ParetoEmbedderSGD(NeuralNet):
    def __init__(
        self,
        *args,
        module=ParetoEmbedding,
        criterion=ParetoEmbeddingLoss,
        optimizer=None,
        callbacks=None,
        **kwargs
    ):
        if optimizer is None:
            optimizer = torch.optim.SGD
        super().__init__(
            *args, module=module, criterion=criterion, optimizer=optimizer, **kwargs
        )

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self.criterion_(y_pred, y_true, X=X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return instance_informedness_np(y_pred, y)

    def predict(self, X):
        scores = super().predict(X)
        return all_pareto_fronts(scores)


class ParetoEmbedderSGDSoft(NeuralNet):
    def __init__(
        self,
        *args,
        module=ParetoEmbedding,
        criterion=ParetoEmbeddingLossSoft,
        optimizer=None,
        callbacks=None,
        **kwargs
    ):
        if optimizer is None:
            optimizer = torch.optim.SGD
        super().__init__(
            *args, module=module, criterion=criterion, optimizer=optimizer, **kwargs
        )

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self.criterion_(y_pred, y_true, X=X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return instance_informedness_np(y_pred, y)

    def predict(self, X):
        scores = super().predict(X)
        return all_pareto_fronts(scores)


class FATEParetoEmbedder(ParetoEmbedder):
    def __init__(
        self,
        *args,
        module=FATE,
        set_encoder="fspool",
        module__set_encoder=None,
        **kwargs
    ):
        if set_encoder in ENCODERS:
            if module__set_encoder is None:
                encoder = ENCODERS[set_encoder]
            else:
                encoder = module__set_encoder
        else:
            raise ValueError("There is no encoder with the name {}".format(set_encoder))
        super().__init__(*args, module=module, module__set_encoder=encoder, **kwargs)


class PWParetoEmbedder(NeuralNet):
    def __init__(
        self,
        *args,
        module=PairwiseEncoder,
        optimizer=None,
        criterion=ParetoEmbeddingLoss,
        **kwargs
    ):
        if optimizer is None:
            optimizer = torch.optim.SGD
        super().__init__(
            *args, module=module, criterion=criterion, optimizer=optimizer, **kwargs
        )

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self.criterion_(y_pred, y_true, X=X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return instance_informedness_np(y_pred, y)

    def predict(self, X):
        scores = super().predict(X)
        return all_pareto_fronts(scores)


class PWParetoEmbedderSoft(NeuralNet):
    def __init__(
        self,
        *args,
        module=PairwiseEncoder,
        optimizer=None,
        criterion=ParetoEmbeddingLossSoft,
        **kwargs
    ):
        if optimizer is None:
            optimizer = torch.optim.SGD
        super().__init__(
            *args, module=module, criterion=criterion, optimizer=optimizer, **kwargs
        )

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self.criterion_(y_pred, y_true, X=X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return instance_informedness_np(y_pred, y)

    def predict(self, X):
        scores = super().predict(X)
        return all_pareto_fronts(scores)
