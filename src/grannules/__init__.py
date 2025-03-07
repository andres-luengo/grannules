import importlib_resources as resources
from grannules.neural_net import *

files = resources.files(__name__)
builtin_model = files / "nn.pkl"

__version__ = "0.0.-1"

del resources

# onnx