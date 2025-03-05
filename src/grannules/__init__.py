import importlib_resources as resources
from grannules.neural_net import *

files = resources.files(__name__)
builtin_model = files / "nn.pkl"

del resources

# onnx