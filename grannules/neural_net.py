from .utils.datatransform import DefaultXTransformer, DefaultyTransformer
from .utils.model import model_from_params, model_from_trial

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
from pathlib import Path
from shutil import rmtree

# since the architecture is a class generated per trial i'll use dill instead of pickle
import dill
import pickle
import json

from typing import Callable

from importlib_resources import files

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


def _split_data(df, random_state = None):
    # preprocessing
    df["log_P"] = np.log(df["P"])
    df["e_log_P"] = df["e_P"] / df["P"]


    log_densities = np.log10(df['M']/df['R']**3)
    log_density_bins = np.digitize(log_densities, bins=np.quantile(log_densities, [0, 0.25, 0.5, 0.75]))
    
    df_train, df_test = train_test_split(df, test_size=0.05, random_state=random_state, stratify=log_density_bins)

    return df_train, df_test

# TODO: function that makes a net "from scratch" since right now you need to pass in a study
# i.e. implement optuna.ipynb
class NNPredictor():    
    """
    Initializes a `NNPredictor`.

    .. note::
        The user shouldn't call this directly, but rather use 
        :meth:`NNPredictor.from_study` or :meth:`NNPredictor.from_pickle`.

    :param train_data: The training data to train the model on in `_train_net`.
        Defaults to ``None``.
    :type train_data: pandas.DataFrame, optional
    :param test_data: The testing data to train the model on in `_train_net`.
        Defaults to ``None``.
    :type test_data: pandas.DataFrame, optional
    :param features: The features to train the model on. 
        Defaults to :attr:`NNPredictor.DEFAULT_FEATURES`.
    :type features: list[str], optional
    :param e_features: The uncertainties of the features. 
        Defaults to ``features`` with an ``'e_'`` prefix on each element.
    :type e_features: list[str], optional
    :param targets: The targets to train the model on. 
        Defaults to :attr:`NNPredictor.DEFAULT_TARGETS`.
    :type targets: list[str], optional
    :param e_targets: The uncertainties of the targets. 
        Defaults to ``targets`` with an ``'e_'`` prefix on each element.
    :type e_targets: list[str], optional
    :param X_transformer: The transformer to use for the features. Calls 
        ``fit_transform`` on the training data, and ``transform`` when using 
        :meth:`predictor.predict`. Defaults to 
        :class:`neural_net.DefaultXTransformer`.
    :type X_transformer: object, optional
    :param y_transformer: The transformer to use for the targets. Calls 
        ``fit_transform`` on the training data, ``transform`` when training, 
        and ``inverse_transform`` when predicting. Defaults to 
        :class:`neural_net.DefaultyTransformer`.
    :type y_transformer: object, optional
    :param nn_state: The state of the neural network. Defaults to ``None``.
    :type nn_state: train_state.TrainState, optional
    :param random_state: The random state to use for training the net. 
        Defaults to ``None``.
    :type random_state: int, optional
    """

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
            X_transformer                               = None,
            y_transformer                               = None,
            state           : train_state.TrainState    = None,
            trial           : optuna.Trial              = None,
            params          : dict                      = None,
            random_state    : int                       = None
    ):
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

        self.X_transformer = X_transformer or DefaultXTransformer()
        self.y_transformer = y_transformer or DefaultyTransformer()
    
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
        
        model = model_from_trial(trial, self.n_outputs)
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
        Train a new neural network model using Optuna for hyperparameter 
        optimization. This method allows training a neural network model by
        either creating a new Optuna study or using an existing one. It 
        supports splitting data into training and testing sets, and optimizing
        the model's hyperparameters through Optuna's study framework.
        
        :param study_or_path: Either an Optuna study object or a string path to
            the study's storage.
        :type study_or_path: optuna.study.Study | str
        :param data: The complete dataset to be split into training and testing
            sets. If provided, `train_data` and `test_data` will be ignored.
        :type data: pandas.DataFrame | None
        :param train_data: Pre-split training data. Used if `data` is not 
            provided.
        :type train_data: pandas.DataFrame | None
        :param test_data: Pre-split testing data. Used if `data` is not
            provided.
        :type test_data: pandas.DataFrame | None
        :param study_name: Name of the Optuna study. Required if creating a new
            study.
        :type study_name: str | None
        :param random_state: Random seed for reproducibility in data splitting
            and training.
        :type random_state: int | None
        :param load_study_if_exists: Whether to load an existing study if it
            already exists.
        :type load_study_if_exists: bool
        :param pruner: Optuna pruner to use for early stopping during
            optimization.
        :type pruner: optuna.pruners.BasePruner
        :param study_kwargs: Additional keyword arguments for creating the 
            Optuna study.
        :type study_kwargs: dict
        :param n_trials: Number of trials to run for hyperparameter 
            optimization.
        :type n_trials: int
        :param optuna_kwargs: Additional keyword arguments for the
            `study.optimize` method.
        :type optuna_kwargs: dict
        :param kwargs: Additional keyword arguments for the neural network
            predictor initialization.
        :type kwargs: dict
        :returns: A tuple containing the trained neural network predictor and
            the Optuna study.
        :rtype: tuple[NNPredictor, optuna.study.Study]
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
            train_data, test_data = _split_data(data, random_state=random_state)
        
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
        Creates an `NNPredictor` from the best trial in an Optuna study.

        :param study_or_path: Union[optuna.study.Study, str]
            An Optuna study or a path to an Optuna database.
            If a path is provided, the study is loaded from the database, and `study_name` must be specified.
        :param data: Optional[pandas.DataFrame]
            The complete dataset to train the model. It will be split using `neural_net.split_data()`.
            Required if `train_data` and `test_data` are not provided.
        :param train_data: Optional[pandas.DataFrame]
            The training dataset. Required if `data` is not provided.
        :param test_data: Optional[pandas.DataFrame]
            The testing dataset. Required if `data` is not provided.
        :param study_name: Optional[str]
            The name of the study to load from the database. This is required if `study_or_path` is a path.
            Ignored if `study_or_path` is an Optuna study.
        :param random_state: Optional[int]
            The random state to use for splitting the data and training the neural network.
        :param kwargs: dict
        :return: NNPredictor
            An instance of `NNPredictor` initialized with the best trial from the study.
        """

        
        # TODO: add support for pathlib.Path (overkill)
        if isinstance(study_or_path, str):
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_or_path}")
        elif isinstance(study_or_path, optuna.study.Study):
            study = study_or_path

        if data is not None:
            train_data, test_data = _split_data(data, random_state=random_state)

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

    # @staticmethod
    # def from_pickle(path, **kwargs) -> 'NNPredictor':
    #     with open(path, 'rb') as f:
    #         predictor : NNPredictor = dill.load(f, **kwargs)
    #     state_dict = from_state_dict(train_state.TrainState, predictor._state_dict)
    #     predictor.state = train_state.TrainState.create(
    #         apply_fn=_model_from_trial(predictor.trial, predictor.n_outputs).apply,
    #         params=state_dict["params"],
    #         tx=optax.adam(1e-3) # probably shouldn't train it, but...
    #     )
    #     return predictor

    # @staticmethod
    # def from_pickle2(path, **kwargs) -> 'NNPredictor':
    #     import pickle
    #     with open(path, 'rb') as f:
    #         predictor : NNPredictor = pickle.load(f, **kwargs)
    #     state_dict = from_state_dict(train_state.TrainState, predictor._state_dict)
    #     predictor.state = train_state.TrainState.create(
    #         apply_fn=_model_from_trial(predictor.trial, predictor.n_outputs).apply,
    #         params=state_dict["params"],
    #         tx=optax.adam(1e-3) # probably shouldn't train it, but...
    #     )
    #     return predictor
    
    # # i don't think i can do this without some big rewriting???
    # @classmethod
    # def from_orbax(cls, directory : str, n_outputs : int = None, **kwargs):
    #     checkpoint = from_state_dict(PyTreeCheckpointer().restore(directory))
    #     if n_outputs is None:
    #         n_outputs = len(self.DEFAULT_TARGETS)

    #     state = train_state.TrainState.create(
    #         apply_fn = _model_from_trial(checkpoint['trial'], n_outputs),
    #         params = checkpoint["state"]["params"],
    #         tx = optax.adam(1e-3) # idk that's what i did over the summer
    #     )
    #     predictor = cls(
    #         state = state,
    #         trial = checkpoint['trial'],
    #         **kwargs
    #     )
    
    # def to_orbax(self, directory : str):
    #     """Stores self.state into directory using an orbax PyTreeCheckpointer

    #     Args:
    #         directory (str): Directory passed to checkpointer.
    #     """
    #     checkpoint = {
    #         "state" : self.state,
    #         "trial" : self.trial # saves hyperparams
    #     }
    #     PyTreeCheckpointer().save(directory, checkpoint)
    
    # def to_pickle(self, path, keep_train_data = False, **kwargs):
    #     predictor = copy.deepcopy(self)

    #     if not keep_train_data:
    #         del predictor.train_data
    #         del predictor.test_data

    #     predictor._state_dict = to_state_dict(predictor.state)
    #     del predictor.state

    #     print(vars(predictor).keys())

    #     with open(path, 'wb') as f:
    #         dill.dump(predictor, f, **kwargs)
    
    # def to_pickle2(self, path, keep_train_data = False, **kwargs):
    #     import pickle
    #     predictor = copy.deepcopy(self)

    #     if not keep_train_data:
    #         if hasattr(predictor, "train_data"): del predictor.train_data
    #         if hasattr(predictor, "test_data"): del predictor.test_data

    #     predictor._state_dict = to_state_dict(predictor.state)
    #     del predictor.state

    #     print(vars(predictor).keys())

    #     with open(path, 'wb') as f:
    #         pickle.dump(predictor, f, **kwargs)
    
    def predict(self, X : pd.DataFrame, to_df = False) -> np.ndarray:
        r"""Predicts the parameters :math:`H,\, P,\, \tau,` and 
        :math:`\alpha` for red giant stars using a pre-trained neural network.

        :param X: A pandas DataFrame with columns 'M', 'R', 'Teff', 'FeH', 
            'KepMag', and 'phase'. 

            * 'M': Mass of the star in solar masses.
            * 'R': Radius of the star in solar radii.
            * 'Teff': Effective temperature of the star in Kelvin.
            * 'FeH': Metallicity of the star.
            * 'KepMag': Apparent magnitude of the star in the Kepler band.
            * 'phase': Phase of the star.

        :type X: pandas.DataFrame
        :param to_df: If True, returns the predictions as a pandas DataFrame. 
            Otherwise, returns a NumPy array.
        :type to_df: bool
        :return: Predicted values for :math:`H,\, P,\, \tau,\,` and :math:`\alpha`.
            If `to_df` is True, the result is a pandas DataFrame with columns 
            ['H', 'P', 'tau', 'alpha']. Otherwise, it is a NumPy array.
        :rtype: pandas.DataFrame or numpy.ndarray
        """
        X_ = self.X_transformer.transform(X)
        y_ = self.state.apply_fn(self.state.params, X_, training=False)
        y = self.y_transformer.inverse_transform(y_)
        y_df = pd.DataFrame(y, columns = self.DEFAULT_TARGETS)
        return y_df if to_df else y

    def serialize(self, path: str | Path = None, overwrite: bool = False):
        """
        Serialize the neural network model, its parameters, and data transformations 
        to the specified directory.
        
        This method saves the model's parameters, state, and data transformation 
        details into a directory for later use. If the directory already exists, 
        it can optionally overwrite it.

        :param path: The directory path where the model will be serialized. 
                     Defaults to the current working directory with the name 
                     "grannules-predictor".
        :type path: str | pathlib.Path, optional
        :param overwrite: Whether to overwrite the directory if it already exists. 
                          Defaults to False.
        :type overwrite: bool
        :raises RuntimeError: If attempting to overwrite the current working 
                              directory, root, or home.
        :raises FileExistsError: If the directory already exists and `overwrite` 
                                 is set to False.
        """
        
        if path is None: path = Path.cwd() / "grannules-predictor"
        path = Path(path) # convert to Path if not already
        if path.exists():
            if overwrite:
                if path in {Path.cwd(), Path("/"), Path.home()}:
                    raise RuntimeError(
                        f"Refusing to overwrite {path}"
                    )
                else:
                    rmtree(path) # scary
            else:
                raise FileExistsError(
                    f"{path} already exists. Set overwrite = True to replace "
                    "this."
                )
        path.mkdir()

        params_path = path / "params.json"
        state_path = path / "state.pkl"
        data_transform_path = path / "transform.npy"

        with open(params_path, "w") as f:
            json.dump(self.trial.params, f)
        
        with open(state_path, "wb") as f:
            pickle.dump(to_state_dict(self.state), f)
        
        transform_dict = {
            "X_scale" : self.X_transformer.scale_,
            "X_center" : self.X_transformer.center_,
            "y_scale" : self.y_transformer.scale_,
            "y_center" : self.y_transformer.center_
        }
        with open(data_transform_path, "wb") as f:
            jnp.save(f, transform_dict)
    
    @classmethod
    def deserialize(cls, path: str | Path = None):
        """
        Deserialize a neural network from a directory. 

        This method reads in the same format as :meth:`NNPredictor.serialize`

        :param path: Path to the directory containing the serialized model files. 
                 The directory should include:

                 * params.json: JSON file with model parameters.
                 * state.pkl: Pickle file with the model's state dictionary.
                 * transform.npy: Numpy file with transformation parameters 
                   for input and output scaling.

        :type path: str or Path
        :return: An instance of `NNPredictor` initialized with the deserialized 
            model, state, and transformers.
        :rtype: NNPredictor
        """
        if path is None: path = Path.cwd() / "grannules-predictor"
        path = Path(path)
        
        params_path = path / "params.json"
        state_path = path / "state.pkl"
        data_transform_path = path / "transform.npy"

        with open(params_path, "r") as f:
            params = json.load(f)
        model = model_from_params(params, len(NNPredictor.DEFAULT_TARGETS))
        blank_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(
                jax.random.PRNGKey(0), 
                jax.random.normal(jax.random.PRNGKey(42), (1, len(NNPredictor.DEFAULT_FEATURES))), 
                training=False), 
                tx = optax.adam(1e-3)
        )
        with open(state_path, "rb") as f:
            state_dict = pickle.load(f)

        state = from_state_dict(blank_state, state_dict)

        with open(data_transform_path, "rb") as f:
            transform_dict = jnp.load(f, allow_pickle = True).item()
        # print(f"{transform_dict = !r}")
        X_transformer = DefaultXTransformer(transform_dict["X_center"], transform_dict["X_scale"])
        y_transformer = DefaultyTransformer(transform_dict["y_center"], transform_dict["y_scale"])

        return NNPredictor(
            model_name = "default",
            state = state,
            X_transformer = X_transformer,
            y_transformer = y_transformer
        )

    @classmethod
    def _default_from_serialize(
            cls,
            path = files(__name__) / "../data/default-serialized"
    ):
        return cls.deserialize(path)

    _default_predictor = None
    @classmethod
    def get_default_predictor(cls, *args, **kwargs):
        """Loads a pre-trained NNPredictor singleton.

        :return: A pre-trained NNPredictor
        :rtype: NNPredictor
        """
        if cls._default_predictor is None:
            cls._default_predictor = cls._default_from_serialize(*args, **kwargs)

        return cls._default_predictor

def predict(X: pd.DataFrame, to_df: bool, *args, **kwargs) -> np.ndarray:
    r"""Predicts the parameters :math:`H,\, P,\, \tau,` and 
    :math:`\alpha` for red giant stars using a pre-trained neural network.

    :param X: A pandas DataFrame with columns 'M', 'R', 'Teff', 'FeH', 
        'KepMag', and 'phase'. 
        - 'M': Mass of the star in solar masses.
        - 'R': Radius of the star in solar radii.
        - 'Teff': Effective temperature of the star in Kelvin.
        - 'FeH': Metallicity of the star.
        - 'KepMag': Apparent magnitude of the star in the Kepler band.
        - 'phase': Phase of the star.
    :type X: pandas.DataFrame
    :param to_df: If True, returns the predictions as a pandas DataFrame. 
        Otherwise, returns a NumPy array.
    :type to_df: bool
    :return: Predicted values for :math:`H,\, P,\, \tau,\,` and :math:`\alpha`.
        If `to_df` is True, the result is a pandas DataFrame with columns 
        ['H', 'P', 'tau', 'alpha']. Otherwise, it is a NumPy array.
    :rtype: pandas.DataFrame or numpy.ndarray
    """
    # TODO: MAKE SURE THIS IS WHAT KEPMAG IS???
    predictor = NNPredictor.get_default_predictor(*args, **kwargs)
    return predictor.predict(X, to_df)

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


