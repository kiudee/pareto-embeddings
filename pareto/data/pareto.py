import numpy as np
import pygmo as pg

from pareto.data.dataset import AbstractGenerator
from pareto.util import all_pareto_fronts

__all__ = ["TwoParabola", "DTLZ", "ZDT"]


def _two_parabola(x):
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    return np.concatenate(
        [(x1 - 1) ** 2 + (x2 - 1) ** 4, (x1 + 1) ** 2 + (x2 + 1) ** 4], 1
    )


class TwoParabola(AbstractGenerator):
    def __init__(self, *args, n_features=2, n_instances=1000, n_objects=10, **kwargs):
        super().__init__(
            *args,
            n_features=n_features,
            n_instances=n_instances,
            n_objects=n_objects,
            **kwargs
        )

    def get_xy(self, **kwargs):
        if self.n_features != 2:
            raise ValueError(
                "The two parabola problem is only defined for n_features=2."
            )
        X = self.random_state.uniform(-2, 2, size=(self.n_instances, self.n_objects, 2))
        X_trans = np.empty((self.n_instances, self.n_objects, 2))
        for i in range(len(X)):
            X_trans[i] = _two_parabola(X[i])
        Y = all_pareto_fronts(-X_trans)
        X = X.astype(np.float32)
        Y = Y.astype(bool)
        return X, Y


class DTLZ(AbstractGenerator):
    def __init__(
        self,
        *args,
        n_features=5,
        n_instances=1000,
        n_objects=10,
        n_objectives=3,
        prob_id=1,
        density=100,
        batch_size=1000,
        **kwargs
    ):
        """Constructs a multi-objective box-constrained problem from the DTLZ testsuite

        Parameters
        ----------
        args :
        n_features : int, default=5
        n_instances : int, default=1000
        n_objects : int, default=10
        n_objectives : int, default=3
        prob_id : int in {1, 2, ..., 7}, default=1
        density : int, default=100
            Density of solutions for prob_id = 4
        batch_size : int, default=1000
            The pygmo generator slows down drastically with higher number of instances.
            Thus we let it generate batch_size many instances at a time.
        kwargs :
        """
        super().__init__(
            *args,
            n_features=n_features,
            n_instances=n_instances,
            n_objects=n_objects,
            **kwargs
        )
        if prob_id not in range(1, 8):
            raise ValueError("prob_id has to be one of {1..7}")
        self.prob_id = prob_id
        self.density = density
        if n_objectives < 2 or self.n_features <= n_objectives:
            raise ValueError(
                "The inequality 2 <= n_objectives < n_features is violated."
            )
        self.n_objectives = n_objectives
        self.problem = pg.dtlz(
            prob_id=self.prob_id,
            fdim=self.n_objectives,
            dim=self.n_features,
            alpha=self.density,
        )
        self.batch_size = batch_size

    def get_xy(self, **kwargs):
        n_rounds = int(np.ceil(self.n_instances / self.batch_size))
        remaining_instances = self.n_instances
        X = []
        Y = []
        for _ in range(n_rounds):
            instances = min(remaining_instances, self.batch_size)
            remaining_instances -= instances
            pop = pg.population(
                prob=self.problem,
                size=instances * self.n_objects,
                seed=int(self.random_state.integers(0, 2 ** 32 - 1)),
            )
            x = pop.get_x().reshape(instances, self.n_objects, self.n_features)
            x = x.astype(np.float32)
            y_raw = pop.get_f().reshape(instances, self.n_objects, self.n_objectives)
            y = all_pareto_fronts(y_raw)
            X.append(x)
            Y.append(y)
        return np.concatenate(X), np.concatenate(Y)


class ZDT(AbstractGenerator):
    def __init__(
        self,
        *args,
        n_features=5,
        n_instances=1000,
        n_objects=10,
        prob_id=1,
        batch_size=1000,
        **kwargs
    ):
        """Constructs a multi-objective box-constrained problem from the ZDT testsuite

        Parameters
        ----------
        args :
        n_features : int, default=30
            Will be used for all problems, except for prob_id=5 where it’s always 175
        n_instances : int, default=1000
        n_objects : int, default=10
        prob_id : int in {1, 2, ..., 7}, default=1
        batch_size : int, default=1000
            The pygmo generator slows down drastically with higher number of instances.
            Thus we let it generate batch_size many instances at a time.
        kwargs :
        """
        super().__init__(
            *args,
            n_features=n_features,
            n_instances=n_instances,
            n_objects=n_objects,
            **kwargs
        )
        if prob_id not in range(1, 7):
            raise ValueError("prob_id has to be one of {1..6}")
        if prob_id == 5:
            if self.n_features % 5 == 1 or self.n_features < 35:
                raise ValueError(
                    "For prob_id=5 the number of features has to be divisible by 5 and >= 35."
                )
        self.prob_id = prob_id
        if self.prob_id != 5:
            self.problem = pg.zdt(prob_id=self.prob_id, param=self.n_features,)
        else:
            self.problem = pg.zdt(
                prob_id=self.prob_id, param=(self.n_features - 25) // 5
            )
        self.batch_size = batch_size

    def get_xy(self, **kwargs):
        n_rounds = int(np.ceil(self.n_instances / self.batch_size))
        remaining_instances = self.n_instances
        X = []
        Y = []
        for _ in range(n_rounds):
            instances = min(remaining_instances, self.batch_size)
            remaining_instances -= instances
            pop = pg.population(
                prob=self.problem,
                size=instances * self.n_objects,
                seed=int(self.random_state.integers(0, 2 ** 32 - 1)),
            )
            x = pop.get_x().reshape(instances, self.n_objects, self.n_features)
            x = x.astype(np.float32)
            y_raw = pop.get_f().reshape(instances, self.n_objects, 2)
            y = all_pareto_fronts(y_raw)
            X.append(x)
            Y.append(y)
        return np.concatenate(X), np.concatenate(Y)
