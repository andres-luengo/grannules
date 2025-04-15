import flax.linen as nn

def model_from_params(hyperparams, n_outputs):
    """Creates a neural network model (:class:`flax.nn.Module`) using the hyperparameters in `hyperparams`.

    :param hyperparams: A dictionary with the following entries:

                        * `'num_layers'`, `int`, the number of layers the model
                        should have.
                        * `'dropout_rate'`, `float`, the fraction of neurons to
                        disable at random on any training run.
                        * `'use_dropout_rate'`, `bool`, whether to use a dropout
                        rate at all.
                        * `'layer_[n]_size'`, `int`, the amount of neurons in
                        the nth layer. There should be one of these arguments
                        for every n in [0, num_layers).
                        * `'layer_[n]_type'`, `str 'relu', 'sigmoid', or 'tanh'` the
                        operation performed by neurons in this layer. There
                        should be one of these entries for every n in [0,
                        num_layers), like for `'layer_[n]_size'`.
    
    :type hyperparams: dict[str, int | float | str]
    :param n_outputs: The number of outputs this neural network should have.
    :type n_outputs: int
    :return: Instance of created model (a subclass of :class:`flax.nn.Module`).
    :rtype: flax.nn.Module
    """
    num_layers = hyperparams['num_layers']

    use_dropout_layer = hyperparams['use_dropout_rate']
    if use_dropout_layer:
        dropout_rate = hyperparams['dropout_rate']
    class StellarModel(nn.Module):
        @nn.compact
        def __call__(self, x, training : bool):
            for i in range(num_layers):
                layer_size = hyperparams[f'layer_{i}_size']
                x = nn.Dense(layer_size)(x)
                layer_type = hyperparams[f'layer_{i}_type']
                match layer_type:
                    case 'relu':
                        x = nn.relu(x)
                    case 'sigmoid':
                        x = nn.sigmoid(x)
                    case 'tanh':
                        x = nn.tanh(x)
                if use_dropout_layer:
                    x = nn.Dropout(rate=dropout_rate)(x, deterministic=not training)

            x = nn.Dense(n_outputs)(x)
            return x
    return StellarModel()

def model_from_trial(trial, n_outputs):
    """
    Does the same thing as :meth:`model_from_params`, but calls the
    `suggest_` methods on trial to gather the hyperparameters.
    :meth:`suggest_categorical` for the `bool` and `str` arguments, and
    :meth:`suggest_int` and :meth:`suggest_float` for the `int` and `float`
    arguments respectively.
    """
    num_layers = trial.suggest_int('num_layers', 1, 5)

    use_dropout_layer = trial.suggest_categorical('use_dropout_rate', [True, False])
    if use_dropout_layer:
        dropout_rate = trial.suggest_float('dropout_rate', 0.001, 0.25, log=True)
    class StellarModel(nn.Module):
        @nn.compact
        def __call__(self, x, training : bool):
            for i in range(num_layers):
                layer_size = trial.suggest_int(f'layer_{i}_size', 16, 512, log=False)
                x = nn.Dense(layer_size)(x)
                layer_type = trial.suggest_categorical(f'layer_{i}_type', ['relu', 'sigmoid', 'tanh'])
                match layer_type:
                    case 'relu':
                        x = nn.relu(x)
                    case 'sigmoid':
                        x = nn.sigmoid(x)
                    case 'tanh':
                        x = nn.tanh(x)
                if use_dropout_layer:
                    x = nn.Dropout(rate=dropout_rate)(x, deterministic=not training)

            x = nn.Dense(n_outputs)(x)
            return x
    return StellarModel()
