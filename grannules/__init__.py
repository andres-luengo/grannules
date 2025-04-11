import importlib_resources as resources

from grannules.neural_net import (
    NNPredictor, predict
)

__version__ = "0.0.-1"

files = resources.files(__name__)
builtin_model = files / "nn.pkl"

del resources