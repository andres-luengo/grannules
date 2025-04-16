r"""
Provides :class:`DataTransformer`, an abstract base class for defining custom
data transformers for data pre-processing, as well as 
:class:`DefaultXTransformer` and :class:`DefaultyTransformer`, the transformers
used by :func:`grannules.NNPredictor.get_default_predictor` and 
:func:`grannules.predict`
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

class DataTransformer(metaclass = ABCMeta):
    r"""
    Custom data transformers for use in creating custom NNPredictors should
    inherit this class and override :meth:`fit_transform`, :meth:`transform`,
    and :meth:`inverse_transform`. Provides default :meth:`__init__` and
    :meth:`fit` methods that may optionally be overridden as well.
    """
    def __init__(self, center: jnp.ndarray, scale: jnp.ndarray):
        self.center_ = center
        self.scale_ = scale
    
    @abstractmethod
    def fit_transform(self, X): pass

    # freebie
    def fit(self, X) -> None:
        r"""
        Calls :meth:`fit_transform` without returning.

        :param X: Passed through to :meth:`fit_transform`
        """
        self.fit_transform(X)

    @abstractmethod
    def transform(self, X): pass
    
    @abstractmethod
    def inverse_transform(self, X): pass

class DefaultXTransformer(DataTransformer):
    def fit_transform(self, X: pd.DataFrame) -> jnp.ndarray:
        X_ = X.copy().values
        X_[:, :3] = jnp.log(X_[:, :3])
        self.center_ = jnp.median(X_, 0)
        self.scale_ = jnp.percentile(X_, 75, 0) - jnp.percentile(X_, 25, 0)
        X_ = (X_ - self.center_) / self.scale_
        return X_
    
    def transform(self, X: pd.DataFrame) -> jnp.ndarray:
        X_ = X.copy().values
        if X_.dtype != np.float64:
            print(X_.dtype)
        X_[:, :3] = jnp.log(X_[:, :3])
        X_ = (X_ - self.center_) / self.scale_
        return X_

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        X_ = X * self.scale_ + self.center_
        X_[:, :3] = jnp.exp(X_[:, :3])
        return X_
    
class DefaultyTransformer(DataTransformer):
    def fit_transform(self, y: pd.DataFrame) -> jnp.ndarray:
        y_ = y.copy().values
        y_ = jnp.log(y_)
        self.center_ = jnp.median(y_, 0)
        self.scale_ = jnp.percentile(y_, 75, 0) - jnp.percentile(y_, 25, 0)
        y_ = (y_ - self.center_) / self.scale_
        return y_
    
    def transform(self, y: pd.DataFrame) -> jnp.ndarray:
        y_ = y.copy().values
        y_ = jnp.log(y_)
        y_ = (y_ - self.center_) / self.scale_
        return y_

    def inverse_transform(self, y: jnp.ndarray) -> jnp.ndarray:
        y_ = y * self.scale_ + self.center_
        y_ = jnp.exp(y_)
        return y_