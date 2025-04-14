import jax.numpy as jnp
import numpy as np

class DefaultXTransformer():
    def __init__(self, center, scale):
        self.center_ = center
        self.scale_ = scale
        
    def fit_transform(self, X):
        X_ = X.copy().values
        X_[:, :3] = jnp.log(X_[:, :3])
        self.center_ = jnp.median(X_, 0)
        self.scale_ = jnp.percentile(X_, 75, 0) - jnp.percentile(X_, 25, 0)
        X_ = (X_ - self.center_) / self.scale_
        return X_
    
    def fit(self, X):
        self.fit_transform(X) # cheese
    
    def transform(self, X):
        X_ = X.copy().values
        if X_.dtype != np.float64:
            print(X_.dtype)
        X_[:, :3] = jnp.log(X_[:, :3])
        X_ = (X_ - self.center_) / self.scale_
        return X_

    def inverse_transform(self, X):
        X_ = X * self.scale_ + self.center_
        X_[:, :3] = jnp.exp(X_[:, :3])
        return X_
    
class DefaultyTransformer():
    def __init__(self, center, scale):
        self.center_ = center
        self.scale_ = scale

    def fit_transform(self, y):
        y_ = y.copy().values
        y_ = jnp.log(y_)
        self.center_ = jnp.median(y_, 0)
        self.scale_ = jnp.percentile(y_, 75, 0) - jnp.percentile(y_, 25, 0)
        y_ = (y_ - self.center_) / self.scale_
        return y_
    
    def fit(self, y):
        self.fit_transform(y) # cheese
    
    def transform(self, y):
        y_ = y.copy().values
        y_ = jnp.log(y_)
        y_ = (y_ - self.center_) / self.scale_
        return y_

    def inverse_transform(self, y):
        y_ = y * self.scale_ + self.center_
        y_ = jnp.exp(y_)
        return y_