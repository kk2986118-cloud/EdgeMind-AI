import unittest

from edgemind.runtime import (
    EdgeMindEngine,
    ModelInfo,
    ModelBackend,
)


class FakeBackend(ModelBackend):

    def load(self, model):
        self.loaded = True

    def generate(self, prompt):
        return "Test response from EdgeMind AI"

    def unload(self):
        self.loaded = False


class TestEdgeMindEngine(unittest.TestCase):

    def setUp(self):
        self.engine = EdgeMindEngine(memory_limit_mb=4096)
        self.model = ModelInfo(
            name="test-model",
            backend="fake",
            memory_required_mb=512,
        )
        self.backend = FakeBackend()

    def test_register_model(self):
        self.engine.register_model(self.model)
        self.assertEqual(len(self.engine.models), 1)

    def test_load_and_generate(self):
        self.engine.register_model(self.model)
        self.engine.attach_backend("test-model", self.backend)
        self.engine.load_model("test-model")

        result = self.engine.generate("Hello")

        self.assertEqual(
            result,
            "Test response from EdgeMind AI"
        )

    def test_unload_model(self):
        self.engine.register_model(self.model)
        self.engine.attach_backend("test-model", self.backend)
        self.engine.load_model("test-model")
        self.engine.unload_model("test-model")

        self.assertFalse(self.model.loaded)

    def test_memory_limit(self):
        self.engine.register_model(
            ModelInfo("large-model", "fake", 8192)
        )
        self.engine.attach_backend("large-model", self.backend)

        with self.assertRaises(MemoryError):
            self.engine.load_model("large-model")


if __name__ == "__main__":
    unittest.main()
