"""Workflow configuration schema.

Kept as a plain dataclass (no pydantic) so `core` stays dependency-free.
Backend-specific settings (`dft_settings`, `mlp_settings`, `tps_settings`)
are opaque dicts - each package's factory decides how to interpret its own
slice, `core`/`workflow` never look inside them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GATES = {
    "dft_submit": "manual",
    "smd_run": "manual",
    "tps_run": "manual",
    "continue_next_gen": "auto",
}


@dataclass
class WorkflowConfig:
    root_dir: Path
    initial_structures: Path
    initial_trajectory: Path

    dft_backend: str = "mock"
    mlp_backend: str = "mock"
    tps_backend: str = "mock"

    dft_settings: dict[str, Any] = field(default_factory=dict)
    mlp_settings: dict[str, Any] = field(default_factory=dict)
    tps_settings: dict[str, Any] = field(default_factory=dict)

    committee_seeds: list[int] = field(default_factory=lambda: [123, 234, 345])
    n_tps_trajectories: int = 20
    top_k_frames_per_generation: int = 10
    max_generations: int = 3

    gates: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GATES))

    @classmethod
    def from_json(cls, path: Path) -> "WorkflowConfig":
        payload = json.loads(Path(path).read_text())
        payload["root_dir"] = Path(payload["root_dir"])
        payload["initial_structures"] = Path(payload["initial_structures"])
        payload["initial_trajectory"] = Path(payload["initial_trajectory"])
        merged_gates = dict(DEFAULT_GATES)
        merged_gates.update(payload.get("gates", {}))
        payload["gates"] = merged_gates
        return cls(**payload)

    def to_json(self, path: Path) -> None:
        payload = {
            "root_dir": str(self.root_dir),
            "initial_structures": str(self.initial_structures),
            "initial_trajectory": str(self.initial_trajectory),
            "dft_backend": self.dft_backend,
            "mlp_backend": self.mlp_backend,
            "tps_backend": self.tps_backend,
            "dft_settings": self.dft_settings,
            "mlp_settings": self.mlp_settings,
            "tps_settings": self.tps_settings,
            "committee_seeds": self.committee_seeds,
            "n_tps_trajectories": self.n_tps_trajectories,
            "top_k_frames_per_generation": self.top_k_frames_per_generation,
            "max_generations": self.max_generations,
            "gates": self.gates,
        }
        Path(path).write_text(json.dumps(payload, indent=2))
