"""
EdgeMind AI - Llama.cpp Inference Backend

Loads and runs real GGUF language models using llama-cpp-python.
"""

from pathlib import Path
from typing import Any, Optional
import threading

from llama_cpp import Llama

from edgemind.runtime import ModelBackend, ModelInfo


class LlamaCppBackend(ModelBackend):
    """Real local inference backend powered by llama.cpp."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        if not model_path or not model_path.strip():
            raise ValueError("Model path cannot be empty")

        if n_ctx <= 0:
            raise ValueError("Context size must be positive")

        if max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")

        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.llm: Optional[Llama] = None
        self.lock = threading.Lock()

    def load(self, model: ModelInfo) -> None:
        if self.llm is not None:
            raise RuntimeError("A model is already loaded")

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"GGUF model file not found: {self.model_path}"
            )

        if not self.model_path.suffix.lower() == ".gguf":
            raise ValueError("Only GGUF model files are supported")

        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
        except Exception:
            self.llm = None
            raise

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.llm is None:
            raise RuntimeError("No GGUF model is loaded")

        with self.lock:
            response: Any = self.llm.create_completion(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                echo=False,
            )

        text = response["choices"][0]["text"]

        if not isinstance(text, str):
            raise RuntimeError("Invalid response from inference engine")

        return text.strip()

    def unload(self) -> None:
        with self.lock:
            if self.llm is None:
                return

            self.llm.close()
            self.llm = None
