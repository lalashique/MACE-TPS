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
    "init_md": "manual",  # only checked when init_md_settings is set
    "dft_submit": "manual",
    "smd_run": "manual",
    "tps_run": "manual",
    "continue_next_gen": "auto",
}


@dataclass
class WorkflowConfig:
    root_dir: Path
    initial_structures: Path | list[Path]  # a list -> one DFT structure_N per file (legacy init/coord*.xyz)
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

    # Legacy-workflow parity options (defaults keep the old simple behaviour).
    n_dft_splits: int = 1  # generations > 0: split selected frames into N parallel DFT jobs
    dft_frames_per_split: int = 0  # if > 0, overrides n_dft_splits: N = ceil(frames / this)
    cumulative_training: bool = False  # train/test from ALL generations' labels, not just the latest
    selection_mode: str = "topk"  # "topk": engine ranks by force variance; "sampler": trust the TPS
    #                               backend's own selection (legacy select_frames thresholds)
    smd_settings: dict[str, Any] = field(default_factory=dict)  # see engine._prepare_initial_path
    # Initial-data generation (paper SI): foundation-model MD of every input xyz before gen-0 DFT.
    # Keys: inputs [xyz files with box], model ("mace_mp:medium"), auto_run (bool), + mlp.init_md settings
    # (temperature, time_step, total_steps, n_frames, friction_per_fs, cell, seed, device). See engine._init_md_outputs.
    init_md_settings: dict[str, Any] = field(default_factory=dict)

    gates: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GATES))

    @classmethod
    def from_json(cls, path: Path) -> "WorkflowConfig":
        payload = json.loads(Path(path).read_text())
        payload["root_dir"] = Path(payload["root_dir"])
        init = payload["initial_structures"]
        payload["initial_structures"] = [Path(x) for x in init] if isinstance(init, list) else Path(init)
        payload["initial_trajectory"] = Path(payload["initial_trajectory"])
        merged_gates = dict(DEFAULT_GATES)
        merged_gates.update(payload.get("gates", {}))
        payload["gates"] = merged_gates
        return cls(**payload)

    def to_json(self, path: Path) -> None:
        payload = {
            "root_dir": str(self.root_dir),
            "initial_structures": (
                [str(x) for x in self.initial_structures]
                if isinstance(self.initial_structures, list)
                else str(self.initial_structures)
            ),
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
            "n_dft_splits": self.n_dft_splits,
            "dft_frames_per_split": self.dft_frames_per_split,
            "cumulative_training": self.cumulative_training,
            "selection_mode": self.selection_mode,
            "smd_settings": self.smd_settings,
            "init_md_settings": self.init_md_settings,
            "gates": self.gates,
        }
        Path(path).write_text(json.dumps(payload, indent=2))
