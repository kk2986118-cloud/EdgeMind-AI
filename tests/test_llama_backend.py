import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from edgemind.runtime import ModelInfo
from edgemind.llama_backend import LlamaCppBackend


class TestLlamaCppBackend(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "test-model.gguf"
        self.model_path.touch()

        self.backend = LlamaCppBackend(
            model_path=str(self.model_path),
            n_ctx=2048,
            max_tokens=100,
        )

        self.model = ModelInfo(
            name="test-model",
            backend="llama.cpp",
            memory_required_mb=512,
        )

    def tearDown(self):
        self.backend.unload()
        self.temp_dir.cleanup()

    @patch("edgemind.llama_backend.Llama")
    def test_load_model(self, mock_llama):
        self.backend.load(self.model)

        mock_llama.assert_called_once()
        self.assertIsNotNone(self.backend.llm)

    @patch("edgemind.llama_backend.Llama")
    def test_generate_response(self, mock_llama):
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {
            "choices": [{"text": "Hello from EdgeMind"}]
        }
        mock_llama.return_value = mock_instance

        self.backend.load(self.model)
        result = self.backend.generate("Hello")

        self.assertEqual(result, "Hello from EdgeMind")
        mock_instance.create_completion.assert_called_once()

    def test_generate_without_loading(self):
        with self.assertRaises(RuntimeError):
            self.backend.generate("Hello")

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            self.backend.generate(" ")

    def test_missing_model_file(self):
        backend = LlamaCppBackend(
            model_path=str(Path(self.temp_dir.name) / "missing.gguf")
        )

        with self.assertRaises(FileNotFoundError):
            backend.load(self.model)

    def test_invalid_model_extension(self):
        invalid_path = Path(self.temp_dir.name) / "model.bin"
        invalid_path.touch()

        backend = LlamaCppBackend(model_path=str(invalid_path))

        with self.assertRaises(ValueError):
            backend.load(self.model)

    @patch("edgemind.llama_backend.Llama")
    def test_unload_model(self, mock_llama):
        mock_instance = MagicMock()
        mock_llama.return_value = mock_instance

        self.backend.load(self.model)
        self.backend.unload()

        mock_instance.close.assert_called_once()
        self.assertIsNone(self.backend.llm)

    @patch("edgemind.llama_backend.Llama")
    def test_load_failure(self, mock_llama):
        mock_llama.side_effect = RuntimeError("Model loading failed")

        with self.assertRaises(RuntimeError):
            self.backend.load(self.model)

        self.assertIsNone(self.backend.llm)

    def test_invalid_context_size(self):
        with self.assertRaises(ValueError):
            LlamaCppBackend(
                model_path="test.gguf",
                n_ctx=0,
            )


if __name__ == "__main__":
    unittest.main()
