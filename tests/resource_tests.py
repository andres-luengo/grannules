import unittest
import importlib_resources as resources

class ResourceTests(unittest.TestCase):
    def test_import_grannules(self):
        import grannules
    
    def test_import_pkl(self):
        from grannules import builtin_model
    
    def test_load_pkl(self):
        # this ALMOST works
        # just need to move away from pickle file
        from grannules import builtin_model
        import dill
        with resources.as_file(builtin_model) as path:
            with path.open("rb") as f:
                m = dill.load(f)

if __name__ == "__main__": unittest.main()