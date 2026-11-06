"""MACE-MP foundation models (no fine-tuning): resolve a model spec to a local file / calculator.

Spec: "mace_mp:<small|medium|large|...>" (downloaded & cached by mace-torch under ~/.cache/mace),
or a plain path to a model file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREFIX = "mace_mp:"


@dataclass
class ModelSpec:
    spec: str

    @property
    def is_foundation(self) -> bool:
        return self.spec.startswith(PREFIX)

    def local_path(self) -> Path:
        """Local model file (downloads a foundation model on first use)."""
        if not self.is_foundation:
            p = Path(self.spec)
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            return p
        from mace.calculators.foundations_models import download_mace_mp_checkpoint

        return Path(download_mace_mp_checkpoint(self.spec[len(PREFIX):]))

    def calculator(self, device: str = "cpu", default_dtype: str = "float32"):
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=[str(self.local_path())], device=device, default_dtype=default_dtype)


def resolve_model_spec(spec: str | Path) -> ModelSpec:
    return ModelSpec(str(spec))
