import jax, jax.numpy as jnp, jax.random as random
import orbax.checkpoint as ocp
import os


def train_test_split(key: jax.Array, X: jax.Array, y: jax.Array, split: float = 0.2, shuffle: bool = False):
    """
    Shuffle and split dataset.
    Arguments:
        X      : Array of inputs of shape: (samples, features).
        y      : Array of outputs of shape: (samples, features).
        key    : A jax.random.key to seed the row permutation.
        split  : Fraction of dataset in the TEST sample. So, split=0.1 means that 10% of your samples will be in test.
        shuffle: Whether to shuffle dataset before splitting or not.
    """
    N = X.shape[0]
    if shuffle:
        rows = jnp.arange(N)
        shuffled_rows = random.permutation(key, rows)
        X = X[shuffled_rows, :]
        y = y[shuffled_rows, :]
    idx = int(N*split)
    X_train, X_test = X[idx:, :], X[:idx, :]
    y_train, y_test = y[idx:, :], y[:idx, :]
    return X_train, y_train, X_test, y_test


def mse_loss(predicted, target):
    return  jnp.mean(jnp.square(predicted - target))


@jax.jit
def smoothness_loss(pred):
    # Penalize large differences between consecutive predictions
    diffs = pred[:, 1:] - pred[:, :-1]
    return jnp.mean(diffs**2)


def main_test():
    key = random.key(611)
    x = random.uniform(key=key, shape=(100,7))
    y = random.uniform(key=key, shape=(100,7))
    X_train, y_train, X_test, y_test = train_test_split(key, x, y, split=0.1, shuffle=True)
    assert X_train[0].all() == y_train[0].all()
    assert X_test[0].all() == y_test[0].all()


def save_model(name, state):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                ".."))
    checkpoint = {
        'state': state,
    }
    dir_path = os.path.abspath(f'{project_root}/emulator/checkpoints/{name}')
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(dir_path, checkpoint)


def load_model(name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                ".."))

    dir_path = os.path.abspath(f'{project_root}/emulator/checkpoints/{name}')
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(dir_path)


if __name__ == '__main__':
    main_test()