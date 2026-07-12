from __future__ import annotations

from core.interfaces import MLPBackend


def create_mlp_backend(backend: str, settings: dict) -> MLPBackend:
    if backend == "mock":
        from mlp.mock_backend import MockMLPBackend

        return MockMLPBackend()
    if backend == "mace":
        from mlp.mace_backend import MACEBackend

        return MACEBackend(e0s=settings.get("E0s", {}), config_file=settings.get("config_file"))
    raise ValueError(f"Unknown MLP backend: {backend}")
