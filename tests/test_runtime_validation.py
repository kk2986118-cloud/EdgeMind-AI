import unittest
from edgemind.runtime import EdgeMindEngine, ModelInfo


class TestRuntimeValidation(unittest.TestCase):

    def setUp(self):
        self.engine = EdgeMindEngine(memory_limit_mb=4096)

    def test_duplicate_model_registration(self):
        model = ModelInfo("test", "fake", 512)
        self.engine.register_model(model)
        with self.assertRaises(ValueError):
            self.engine.register_model(model)

    def test_invalid_memory_requirement(self):
        model = ModelInfo("invalid", "fake", 0)
        with self.assertRaises(ValueError):
            self.engine.register_model(model)

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            self.engine.load_model("unknown")

    def test_generate_without_loaded_model(self):
        with self.assertRaises(RuntimeError):
            self.engine.generate("Hello")

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            self.engine.generate("   ")

    def test_missing_backend(self):
        model = ModelInfo("test", "fake", 512)
        self.engine.register_model(model)
        with self.assertRaises(RuntimeError):
            self.engine.load_model("test")


if __name__ == "__main__":
    unittest.main()
