"""Approval gates - the replacement for the legacy code's blocking
`input("Continue? (y/N): ")` calls.

A gate has a *name* (a stage in the pipeline, e.g. "smd_run",
"continue_next_gen") whose mode ("auto" or "manual") is set once in the
workflow config, and a per-occurrence *gate_id* (e.g. "gen0:smd_run") that
gets individually approved.

- "auto" gates never block - fine for CI / the mock smoke test.
- "manual" gates raise WaitingForApproval the first time they're hit. The
  CLI (or, later, a GUI) surfaces the instructions, and a person calls
  `GateController.approve(gate_id)` - typically via `mtps approve <gate_id>`.
  Re-running the workflow re-checks the same gate_id, sees it's approved, and
  continues. This makes the engine non-blocking and resumable instead of
  sitting on a stdin prompt.
"""

from __future__ import annotations

import json
from pathlib import Path


class WaitingForApproval(RuntimeError):
    def __init__(self, gate_name: str, gate_id: str, instructions: str):
        super().__init__(f"[{gate_id}] waiting for approval: {instructions}")
        self.gate_name = gate_name
        self.gate_id = gate_id
        self.instructions = instructions


class GateController:
    APPROVALS_FILENAME = "gate_approvals.json"

    def __init__(self, gates: dict, root_dir: Path):
        self.gates = gates or {}
        self.approvals_path = Path(root_dir) / self.APPROVALS_FILENAME
        self._approved = self._load()

    def _load(self) -> set:
        if self.approvals_path.exists():
            return set(json.loads(self.approvals_path.read_text()))
        return set()

    def _save(self) -> None:
        self.approvals_path.parent.mkdir(parents=True, exist_ok=True)
        self.approvals_path.write_text(json.dumps(sorted(self._approved), indent=2))

    def is_auto(self, gate_name: str) -> bool:
        return self.gates.get(gate_name, "manual") == "auto"

    def approve(self, gate_id: str) -> None:
        self._approved.add(gate_id)
        self._save()

    def is_approved(self, gate_id: str) -> bool:
        return gate_id in self._approved

    def check(self, gate_name: str, gate_id: str, instructions: str) -> None:
        """No-op if the gate is auto or already approved; otherwise raises
        WaitingForApproval for the caller (the workflow engine) to surface.
        """
        if self.is_auto(gate_name) or self.is_approved(gate_id):
            return
        raise WaitingForApproval(gate_name, gate_id, instructions)
