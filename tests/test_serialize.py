import unittest

from grannules import NNPredictor

import importlib_resources as resources
import pandas as pd

class TestSerialize(unittest.TestCase):
    def test_default_serialize(self):
        NNPredictor.get_default_predictor()