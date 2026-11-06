"""Completion check for a CP2K reftraj job (stdlib only, so it is unit-testable
without ase/loguru). Same criterion as the legacy workflow.py: the position and
force trajectories must have as many lines as the input coord.xyz, i.e. CP2K
wrote every frame. Parsing a half-written trajectory would silently give
truncated training data.
"""

from __future__ import annotations

from pathlib import Path


def _lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def reftraj_missing(struct_dir: Path) -> str | None:
    """Return None if the job in `struct_dir` finished, else a short reason."""
    struct_dir = Path(struct_dir)
    coord = struct_dir / "coord.xyz"
    if not coord.exists():
        return "coord.xyz missing"
    n = _lines(coord)
    for name in ("cp2k-pos-1.xyz", "cp2k-frc-1.xyz"):
        f = struct_dir / name
        if not f.exists():
            return f"{name} missing"
        m = _lines(f)
        if m != n:
            return f"{name} has {m} lines, coord.xyz has {n}"
    return None
