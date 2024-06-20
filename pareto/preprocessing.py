import numbers

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


def _handle_zeros_in_scale(scale, copy=True):
    """ Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features."""

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


class StandardScaler3D(StandardScaler):
    def partial_fit(self, X, y=None):
        """
        Online computation of mean and std on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.
        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_objects, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Transformer instance.
        """
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
            ensure_2d=False,
            allow_nd=True,
        )
        if sparse.issparse(X):
            raise ValueError("This preprocessor does not support sparse input yet.")

        if not hasattr(self, "scale_"):
            self.mean_ = 0.0
            if self.with_std:
                self.var_ = 0.0
            else:
                self.var_ = None

        if not self.with_mean and not self.with_std:
            self.mean_ = None
            self.var_ = None
        else:
            X_flat = X.reshape(-1, X.shape[2])
            self.mean_ = np.mean(X_flat, axis=0)
            self.var_ = np.var(X_flat, axis=0)
        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        self.n_samples_seen_ = None
        return self

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(
            X,
            accept_sparse="csr",
            copy=copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
            ensure_2d=False,
            allow_nd=True,
        )

        if sparse.issparse(X):
            raise ValueError("This preprocessor does not support sparse input yet.")
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            raise ValueError("This preprocessor does not support sparse input yet.")
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X
