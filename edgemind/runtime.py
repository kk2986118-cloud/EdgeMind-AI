"""
EdgeMind AI - Core Runtime Engine
Initial implementation
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time


@dataclass
class ModelInfo:
    name: str
    backend: str
    memory_required_mb: int
    loaded: bool = False


class ModelBackend:
    """Interface for connecting real AI inference backends."""

    def load(self, model: ModelInfo) -> None:
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError


class EdgeMindEngine:

    def __init__(self, memory_limit_mb: int = 4096):
        self.memory_limit_mb = memory_limit_mb
        self.models: Dict[str, ModelInfo] = {}
        self.backends: Dict[str, ModelBackend] = {}
        self.active_model: Optional[str] = None
        self.started_at = time.time()

    def register_model(self, model: ModelInfo) -> None:
        if model.name in self.models:
            raise ValueError("Model already registered")

        if model.memory_required_mb <= 0:
            raise ValueError("Invalid memory requirement")

        self.models[model.name] = model

    def attach_backend(
        self, model_name: str, backend: ModelBackend
    ) -> None:
        if model_name not in self.models:
            raise KeyError("Model is not registered")

        self.backends[model_name] = backend

    def load_model(self, model_name: str) -> None:
        if model_name not in self.models:
            raise KeyError("Unknown model")

        model = self.models[model_name]
        backend = self.backends.get(model_name)

        if backend is None:
            raise RuntimeError("No inference backend attached")

        if model.memory_required_mb > self.memory_limit_mb:
            raise MemoryError("Model exceeds configured memory budget")

        if self.active_model:
            self.unload_model(self.active_model)

        backend.load(model)
        model.loaded = True
        self.active_model = model_name

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.active_model is None:
            raise RuntimeError("No model is loaded")

        backend = self.backends[self.active_model]
        return backend.generate(prompt)

    def unload_model(self, model_name: str) -> None:
        model = self.models[model_name]

        if not model.loaded:
            return

        backend = self.backends[model_name]
        backend.unload()

        model.loaded = False
        if self.active_model == model_name:
            self.active_model = None

    def status(self) -> Dict[str, Any]:
        return {
            "engine": "EdgeMind AI",
            "active_model": self.active_model,
            "registered_models": len(self.models),
            "loaded_models": sum(
                model.loaded for model in self.models.values()
            ),
            "memory_limit_mb": self.memory_limit_mb,
            "uptime_seconds": round(
                time.time() - self.started_at, 2
            ),
        }
