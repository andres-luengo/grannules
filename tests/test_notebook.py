import unittest
import json

class TestNotebook(unittest.TestCase):
    def test_notebook(self):
        filename = '../docs/usage.ipynb'
        with open(filename) as fp:
            nb = json.load(fp)

        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                for output in cell['outputs']:
                    self.assertNotEqual(
                        output['output_type'], "error",
                        "Usage notebook has error output."
                    )