import numpy as np
import pandas as pd
import optax

from sklearn.model_selection import train_test_split

from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_state_dict, from_state_dict

import jax 
import jax.numpy as jnp

import optuna

import functools
import copy

# since the architecture is a class generated per trial i'll use dill instead of pickle
import dill

from typing import Callable

quiet = False

# # TODO: maybe possibly reconsider this?
# def augment_data(df, num_extra, features, e_features, targets, e_targets):
#     unaugmented_data = df.copy()
#     for i in range(num_extra):
#         extra_data = pd.DataFrame(columns=unaugmented_data.columns)
#         extra_data.index.name = "KIC"
#         for feature, e_feature in zip(features, e_features):
#             feature_noise = np.random.normal(0, unaugmented_data[e_feature])
#             extra_data[feature] = unaugmented_data[feature] + feature_noise
#             extra_data[e_feature] = unaugmented_data[e_feature]
#         for target, e_target in zip(targets, e_targets):
#             target_noise = np.random.normal(0, unaugmented_data[e_target])
#             extra_data[target] = unaugmented_data[target] + target_noise
#             extra_data[e_target] = unaugmented_data[e_target]
#         extra_data.index = unaugmented_data.index + i*100000000
#         df = pd.concat([df, extra_data])


def split_data(df, random_state = None):
    # preprocessing
    df["log_P"] = np.log(df["P"])
    df["e_log_P"] = df["e_P"] / df["P"]


    log_densities = np.log10(df['M']/df['R']**3)
    log_density_bins = np.digitize(log_densities, bins=np.quantile(log_densities, [0, 0.25, 0.5, 0.75]))
    
    df_train, df_test = train_test_split(df, test_size=0.05, random_state=random_state, stratify=log_density_bins)

    return df_train, df_test

def _model_from_trial(trial, n_outputs):
        """Returns a StellarModel with the architecture defined by `trial`."""
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

class DefaultXTransformer():
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

# TODO: function that makes a net "from scratch" since right now you need to pass in a study
# i.e. implement optuna.ipynb
class NNPredictor():
    DEFAULT_FEATURES = ['M', 'R', 'Teff', 'FeH', 'KepMag', 'phase']
    DEFAULT_TARGETS = ['H', 'P', 'tau', 'alpha']

    def __init__(
            self,
            model_name      : str | None                = None,
            train_data      : pd.DataFrame | None       = None,
            test_data       : pd.DataFrame | None       = None,
            features        : list[str]                 = DEFAULT_FEATURES,
            e_features      : list[str] | None          = None,
            targets         : list[str]                 = DEFAULT_TARGETS,
            e_targets       : list[str] | None          = None,
            X_transformer                               = DefaultXTransformer(),
            y_transformer                               = DefaultyTransformer(),
            state           : train_state.TrainState    = None,
            trial           : optuna.Trial              = None,
            params          : dict                      = None,
            random_state    : int                       = None
    ):
        """
        Initializes a `NNPredictor`. 
        The user shouldn't call use this directly, but rather use `NNPredictor.from_study()` or `NNPredictor.from_pickle()`.
        ## Parameters
        `train_data`: `pd.DataFrame`
            The training data to train the model on in `_train_net`.
            Defaults to `None`.
        `test_data`: `pd.DataFrame`
            The testing data to train the model on in `_train_net`.
            Defaults to `None`.
        `features`: `list[str]`
            The features to train the model on. 
            Defaults to `NNPredictor.DEFAULT_FEATURES`.
        `e_features`: `list[str]`
            The uncertainties of the features. 
            Defaults to `features` with an 'e_' prefix on each element.
        `targets`: `list[str]`
            The targets to train the model on. 
            Defaults to `NNPredictor.DEFAULT_TARGETS`.
        `e_targets`: `list[str]`
            The uncertainties of the targets. 
            Defaults to `targets` with an 'e_' prefix on each element.
        `X_transformer`
            The transformer to use for the features. Calls `fit_transform` on the training data, and
            `transform` when using `predictor.predict()`. 
            Defaults to `neural_net.DefaultXTransformer`.
        `y_transformer`
            The transformer to use for the targets. Calls `fit_transform` on the training data,
            `transform` when training, and `inverse_transform` when predicting. 
            Defaults to `neural_net.DefaultyTransformer`.
        `nn_state`: `train_state.TrainState | None`
            The state of the neural network.
            Defaults to `None`.
        `random_state`: `int | None`
            The random state to use for training the net.
            Defaults to `None`.
        """

        self.model_name = model_name

        self.train_data = train_data
        self.test_data = test_data

        self.state = state
        self._state_dict = None
        self.trial = trial

        self.random_state = random_state or 0

        if not random_state:
            self.random_state = 0
        else:
            self.random_state = random_state

        self.features = features
        if e_features is None:
            self.e_features = ['e_' + feature for feature in features]

        self.targets = targets
        if e_targets is None:
            self.e_targets = ['e_' + target for target in targets]
        self.n_outputs = len(targets)

        self.X_transformer = X_transformer
        self.y_transformer = y_transformer
    
    def _train_net(self, trial):
        X_train = self.X_transformer.fit_transform(self.train_data[self.features])
        X_test = self.X_transformer.transform(self.test_data[self.features])
        y_train = self.y_transformer.fit_transform(self.train_data[self.targets])
        y_test = self.y_transformer.transform(self.test_data[self.targets])
        e_y_train = self.train_data[self.e_targets].values
        e_y_test = self.test_data[self.e_targets].values
        e_X_train = self.train_data[self.e_features].values
        e_X_test = self.test_data[self.e_features].values

        # since we mostly deal with uncertainty squared
        e_X_train2 = (e_X_train ** 2)
        e_X_test2 = (e_X_test ** 2)
        e_y_train2 = (e_y_train ** 2)
        e_y_test2 = (e_y_test ** 2)

        if not quiet:
            print("Training net with trial ", trial.number)
        
        model = _model_from_trial(trial, self.n_outputs)
        config = {
            'warmup_epochs': trial.suggest_int('warmup_epochs', 10, 300, log=True),
            'num_epochs': int(5e4)
        }

        def cosine_learning_schedule(config, base_learning_rate, min_learning_rate, steps_per_epoch, num_cycles):
            """Creates learning rate schedule."""
            warmup_fn = optax.linear_schedule(
                init_value=min_learning_rate, end_value=base_learning_rate,
                transition_steps=config["warmup_epochs"] * steps_per_epoch)
            cosine_epochs = max((config["num_epochs"]) // num_cycles - config["warmup_epochs"], 1)
            cosine_fn = optax.cosine_decay_schedule(
                init_value=base_learning_rate,
                decay_steps=cosine_epochs * steps_per_epoch,
                alpha=min_learning_rate / base_learning_rate)

            schedules = []
            boundaries = []
            for i in range(num_cycles):
                schedules.append(warmup_fn)
                schedules.append(cosine_fn)

                warmup_boundary = (config["warmup_epochs"] * (i + 1) + cosine_epochs * i) * steps_per_epoch
                boundaries.append(warmup_boundary)
                if i < num_cycles - 1:
                    cosine_boundary = (config["warmup_epochs"] * (i + 1) + cosine_epochs * (i + 1)) * steps_per_epoch
                    boundaries.append(cosine_boundary)

            schedule_fn = optax.join_schedules(
                schedules=schedules,
                boundaries=boundaries)
            return schedule_fn

        base_learning_rate = trial.suggest_float('base_learning_rate', 1e-5, 1e-3, log=True)
        min_learning_rate = base_learning_rate * trial.suggest_float('min_learning_fraction', 0.001, 0.1, log=True)
        learning_fn = cosine_learning_schedule(config, base_learning_rate, min_learning_rate, 1, trial.suggest_int('num_cycles', 1, 10))

        # Set up optimizer
        optimizer = optax.adam(learning_fn)

        # Initialize state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(
                jax.random.PRNGKey(0), 
                jax.random.normal(jax.random.PRNGKey(self.random_state), (1, X_train.shape[1])), 
                training=True), 
            tx=optimizer
        )

        def mse_loss(params, inputs, targets_, uncertainties_squared, training):
            predictions_ = model.apply(params, inputs, rngs=jax.random.PRNGKey(0), training=training)
            predictions = self.y_transformer.inverse_transform(predictions_)
            targets = self.y_transformer.inverse_transform(targets_)
            mse = ((predictions - targets) ** 2) / uncertainties_squared
            loss = jnp.mean(mse)
            return loss

        # Define training step
        @functools.partial(jax.jit, static_argnums=2)
        def train_step(
            state               : train_state.TrainState, 
            batch               : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            learning_rate_fn    : Callable
        ):
            inputs, targets, uncertainties = batch
            loss, grads = jax.value_and_grad(mse_loss)(state.params, inputs, targets, uncertainties, True)
            state = state.apply_gradients(grads=grads)
            learning_rate = learning_rate_fn(state.step)
            return state, loss, learning_rate

        min_val_loss = float('inf')
        epoch = 0

        overfit_metrics = {
            "prev_val_loss": float('inf'),
            "periods": 0, # a "period" being how long between each check
            "period_threshold": 10,
            "quit_early" : False
        }

        for epoch in range(epoch, config["num_epochs"]):
            state, _, _ = train_step(state, (X_train, y_train, e_y_train2), learning_fn)
            
            if epoch % 500 == 0 and epoch // 500 > 1:
                val_loss = mse_loss(state.params, X_test, y_test, e_y_test2, training=False)
                
                # Update the best state if the current validation loss is lower
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    if overfit_metrics["periods"] != 0:
                        overfit_metrics["periods"] = 0 # all's good if it figures it out (unlikely)
                    best_state = state
                # overfitting check
                elif val_loss > overfit_metrics["prev_val_loss"]:
                    overfit_metrics["periods"] += 1
                    
                    if overfit_metrics["periods"] >= overfit_metrics["period_threshold"]:
                        overfit_metrics["quit_early"] = True
                        break


                overfit_metrics["prev_val_loss"] = val_loss
                # optuna pruning
                trial.report(min_val_loss, step=epoch)
                if trial.should_prune():
                    if not quiet:
                        print("Pruning trial ", trial.number)
                    raise optuna.TrialPruned()
        return best_state, min_val_loss
    
    @classmethod
    def train_new(
        cls,

        study_or_path : optuna.study.Study | str,
        data          : pd.DataFrame | None       = None,
        train_data    : pd.DataFrame | None       = None,
        test_data     : pd.DataFrame | None       = None,
        study_name    : str | None                = None,
        random_state  : int | None                = None,

        load_study_if_exists : bool               = True,
        pruner        : optuna.pruners.BasePruner = optuna.pruners.NopPruner(),
        study_kwargs  : dict                      = {},
        
        n_trials      : int                       = 100,
        optuna_kwargs : dict                      = {},
        **kwargs
    ) -> tuple['NNPredictor', optuna.study.Study]:
        """
        Trains a new `NNPredictor` using optuna.

        ## Parameters
        `study_or_path`: `optuna.study.Study` or `str`
            An optuna study or a path to an optuna database.
            If a path is given, the study is fetched from the database, so `study_name` must be set.
            If the study or database does not exist, a new study is created.
        `data`: `pd.DataFrame`
            All of the data to train the model on. Split using neural_net.split_data().
            Required if `train_data` and `test_data` are not set.
        `train_data`: `pd.DataFrame`
            The training data to train the model on. Required if `data` is not set.
        `test_data`: `pd.DataFrame`
            The testing data to train the model on. Required if `data` is not set.
        `study_name`: `str`
            The name of the study to load from the database. Required if `study_or_path` is a path.
            Ignored if `study_or_path` is a study.
        `load_study_if_exists`: `bool`
            Whether to load the study if it already exists in the database.
            Defaults to `True`.
        `pruner`: `optuna.pruners.BasePruner`
            The pruner to use for the study.
            Defaults to `optuna.pruners.NopPruner()`.
        `study_kwargs`: `dict`
            Additional keyword arguments to pass to `optuna.create_study()`.
            Defaults to `{}`.
        `n_trials`: `int`
            The number of trials to run.
            Defaults to `100`.
        `optuna_kwargs`: `dict`
            Additional keyword arguments to pass to `study.optimize()`.
        `**kwargs`
            Additional keyword arguments to pass to the `NNPredictor` constructor.
        """

        if isinstance(study_or_path, str):
            study = optuna.create_study(
                study_name=study_name, 
                storage=f"sqlite:///{study_or_path}",
                load_if_exists=load_study_if_exists,
                pruner=pruner,
                **study_kwargs
            )
        elif isinstance(study_or_path, optuna.study.Study):
            study = study_or_path

        if data is not None:
            train_data, test_data = split_data(data, random_state=random_state)
        
        predictor = cls(
            train_data=train_data,
            test_data=test_data,
            random_state=random_state,
            model_name=study_name,
            **kwargs
        )

        study.optimize(lambda trial: predictor._train_net(trial)[1], n_trials=n_trials, **optuna_kwargs)

        predictor.state = predictor._train_net(study.best_trial)[0]
        predictor.params = predictor.state.params
        predictor.trial = study.best_trial

        return predictor, study

    @classmethod
    def from_study(
        cls,
        study_or_path   : optuna.study.Study | str,
        data            : pd.DataFrame | None       = None,
        train_data      : pd.DataFrame | None       = None,
        test_data       : pd.DataFrame | None       = None,
        study_name      : str | None                = None,
        random_state    : int | None                = None,
        **kwargs
    ) -> 'NNPredictor':
        """
        Creates a `NNPredictor` from the best trial in a study.

        ## Parameters
        `study_or_path`: `optuna.study.Study` or `str`
            An optuna study or a path to an optuna database.
            If a path is given, the study is loaded from the database, so `study_name` must be set.
        `data`: `pd.DataFrame`
            All of the data to train the model on. Split using neural_net.split_data(). 
            Required if `train_data` and `test_data` are not set.
        `train_data`: `pd.DataFrame`
            The training data to train the model on. Required if `data` is not set.
        `test_data`: `pd.DataFrame`
            The testing data to train the model on. Required if `data` is not set.
        `study_name`: `str`
            The name of the study to load from the database. Required if `study_or_path` is a path.
            Ignored if `study_or_path` is a study.
        `random_state`: `int`
            The random state to use for splitting the data and training the net.
        `**kwargs`
            Additional keyword arguments to pass to the `NNPredictor` constructor.
        """
        # TODO: add support for pathlib.Path (overkill)
        if isinstance(study_or_path, str):
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_or_path}")
        elif isinstance(study_or_path, optuna.study.Study):
            study = study_or_path

        if data is not None:
            train_data, test_data = split_data(data, random_state=random_state)

        predictor = cls(
            train_data=train_data,
            test_data=test_data,
            random_state=random_state,
            model_name=study_name,
            **kwargs
        )
        #TODO: put this in __init__
        predictor.trial = study.best_trial
        predictor.state = predictor._train_net(study.best_trial)[0]
        predictor.params = predictor.state.params

        return predictor

    @staticmethod
    def from_pickle(path, **kwargs) -> 'NNPredictor':
        with open(path, 'rb') as f:
            predictor : NNPredictor = dill.load(f, **kwargs)
        state_dict = from_state_dict(train_state.TrainState, predictor._state_dict)
        predictor.state = train_state.TrainState.create(
            apply_fn=_model_from_trial(predictor.trial, predictor.n_outputs).apply,
            params=state_dict["params"],
            tx=optax.adam(1e-3) # probably shouldn't train it, but...
        )
        return predictor
    
    def to_pickle(self, path, keep_train_data = False, **kwargs):
        predictor = copy.deepcopy(self)

        if not keep_train_data:
            del predictor.train_data
            del predictor.test_data

        predictor._state_dict = to_state_dict(predictor.state)
        del predictor.state

        print(vars(predictor).keys())

        with open(path, 'wb') as f:
            dill.dump(predictor, f, **kwargs)
    
    def predict(self, X : pd.DataFrame) -> np.ndarray:
        X_ = self.X_transformer.transform(X)
        y_ = self.state.apply_fn(self.state.params, X_, training=False)
        y = self.y_transformer.inverse_transform(y_)
        return y


# # Alternate version that uses trial.params instead. I don't think we need it, but it's here just in case.
# use_dropout_layer = trial.params['use_dropout_rate']
# if use_dropout_layer:
#     dropout_rate = trial.params['dropout_rate']
# num_layers=  trial.params['num_layers']
# class StellarModel(nn.Module):
#     @nn.compact
#     def __call__(self, x, training : bool):
#         for i in range(num_layers):
#             layer_size = trial.params[f'layer_{i}_size']
#             x = nn.Dense(layer_size)(x)
#             layer_type = trial.params[f'layer_{i}_type']
#             match layer_type:
#                 case 'relu':
#                     x = nn.relu(x)
#                 case 'sigmoid':
#                     x = nn.sigmoid(x)
#                 case 'tanh':
#                     x = nn.tanh(x)
#             if use_dropout_layer:
#                 x = nn.Dropout(rate=dropout_rate)(x, deterministic=not training)

#         x = nn.Dense(y_train.shape[1])(x)
#         return x
# model = StellarModel()
# TODO: put these in a more relevant file

NYQUIST = 283.2114


# PSD model per deassis
def damping(nu):
    eta = np.sinc((1 / 2) * (nu / NYQUIST)) ** 2
    return eta

def PSD(nu, nu_max, H, P, tau, alpha, reshape = True):
    if reshape:
        nu = np.reshape(nu, (1, -1))
        
        nu_max = np.reshape(nu_max, (-1, 1))
        H = np.reshape(H, (-1, 1))
        P = np.reshape(P,(-1, 1))
        tau = np.reshape(tau, (-1, 1))
        alpha = np.reshape(alpha, (-1, 1))
    
    eta = damping(nu)
    b = granulation(nu, P, tau, alpha)
    g = excess(nu, nu_max, H)
    # g = 0
    return eta * (b + g)

def excess(nu, nu_max, H):
    FWHM = 0.66 * nu_max ** 0.88
    G = H * np.exp(-(nu - nu_max)**2 / (FWHM ** 2 / (4 * np.log(2))))
    return G

def granulation(nu, P, tau, alpha):
    granulation = (P / (1 + (2 * np.pi * tau * 1e-6 * nu) ** alpha))
    if not np.isfinite(granulation).all():
        return nu * np.inf
    return granulation