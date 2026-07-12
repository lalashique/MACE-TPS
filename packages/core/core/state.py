"""Explicit, inspectable workflow state - replaces the DONE/PAUSE sentinel
files and blocking input() prompts from the legacy workflow.py.

State lives in one JSON file (`workflow_state.json` in the run's root dir).
The CLI (and later a GUI) reads/writes it through WorkflowState; nothing
about progress is inferred from the filesystem layout anymore.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class GenerationStage(str, Enum):
    PENDING = "pending"
    DFT_RUNNING = "dft_running"
    DFT_DONE = "dft_done"
    MLP_RUNNING = "mlp_running"
    MLP_DONE = "mlp_done"
    TPS_RUNNING = "tps_running"
    TPS_DONE = "tps_done"
    SELECTING = "selecting"
    COMPLETE = "complete"
    WAITING_APPROVAL = "waiting_approval"
    FAILED = "failed"


@dataclass
class GenerationRecord:
    index: int
    stage: GenerationStage = GenerationStage.PENDING
    blocked: bool = False
    pending_gate: Optional[str] = None
    pending_instructions: Optional[str] = None
    error: Optional[str] = None
    labeled_file: Optional[str] = None
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    model_paths: list[str] = field(default_factory=list)
    candidate_frames_file: Optional[str] = None
    selected_frames_file: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationRecord":
        d = dict(d)
        d["stage"] = GenerationStage(d["stage"])
        return cls(**d)


@dataclass
class WorkflowState:
    root_dir: Path
    generations: list[GenerationRecord] = field(default_factory=list)
    current_generation: int = 0

    STATE_FILENAME = "workflow_state.json"

    @property
    def state_path(self) -> Path:
        return Path(self.root_dir) / self.STATE_FILENAME

    def generation(self, index: int) -> GenerationRecord:
        for g in self.generations:
            if g.index == index:
                return g
        record = GenerationRecord(index=index)
        self.generations.append(record)
        return record

    def save(self) -> None:
        payload = {
            "root_dir": str(self.root_dir),
            "current_generation": self.current_generation,
            "generations": [g.to_dict() for g in self.generations],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_or_create(cls, root_dir: Path) -> "WorkflowState":
        root_dir = Path(root_dir)
        path = root_dir / cls.STATE_FILENAME
        if path.exists():
            payload = json.loads(path.read_text())
            return cls(
                root_dir=Path(payload["root_dir"]),
                current_generation=payload["current_generation"],
                generations=[
                    GenerationRecord.from_dict(g) for g in payload["generations"]
                ],
            )
        return cls(root_dir=root_dir)
