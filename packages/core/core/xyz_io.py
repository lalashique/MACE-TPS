"""Minimal extended-XYZ reader/writer, stdlib only.

This exists so the *mock* backends (and the smoke test) can run with zero
third-party dependencies - no ase, no numpy. The format written here is a
subset of real extxyz (species/pos/forces columns + an `energy=` key in the
comment line), so real backends that use ase.io under the hood can still
read files produced by the mocks if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Frame:
    symbols: list[str]
    positions: list[tuple[float, float, float]]
    forces: list[tuple[float, float, float]] = field(default_factory=list)
    energy: float | None = None
    info: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def read_frames(path: Path) -> list[Frame]:
    lines = Path(path).read_text().splitlines()
    frames: list[Frame] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        n_atoms = int(line)
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        energy = None
        for token in comment.split():
            if token.startswith("energy="):
                energy = float(token.split("=", 1)[1])
        symbols, positions, forces = [], [], []
        has_forces = False
        for j in range(n_atoms):
            parts = lines[i + 2 + j].split()
            symbols.append(parts[0])
            positions.append(tuple(float(x) for x in parts[1:4]))
            if len(parts) >= 7:
                has_forces = True
                forces.append(tuple(float(x) for x in parts[4:7]))
        frames.append(
            Frame(
                symbols=symbols,
                positions=positions,
                forces=forces if has_forces else [],
                energy=energy,
                info={"comment": comment},
            )
        )
        i += 2 + n_atoms
    return frames


def write_frames(path: Path, frames: list[Frame]) -> None:
    out_lines = []
    for frame in frames:
        out_lines.append(str(frame.n_atoms))
        comment = f"energy={frame.energy}" if frame.energy is not None else ""
        out_lines.append(comment)
        for idx in range(frame.n_atoms):
            x, y, z = frame.positions[idx]
            row = f"{frame.symbols[idx]} {x:.6f} {y:.6f} {z:.6f}"
            if frame.forces:
                fx, fy, fz = frame.forces[idx]
                row += f" {fx:.6f} {fy:.6f} {fz:.6f}"
            out_lines.append(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(out_lines) + "\n")


def append_frames(path: Path, frames: list[Frame]) -> None:
    existing = read_frames(path) if Path(path).exists() else []
    write_frames(path, existing + frames)


# ---------------------------------------------------------------------------
# Generic, format-agnostic block operations.
#
# The workflow engine needs to split/subset/reorder xyz files (train/test
# split, "take these N frame indices for labeling") without caring what's in
# the Properties=/comment line - real backends write full ASE extxyz with
# Lattice=/Properties= metadata that the Frame parser above doesn't attempt
# to understand. These functions only assume the universal xyz shape
# (atom-count line, comment line, N atom lines) and move whole blocks of
# raw text verbatim, so they work identically for mock output and for real
# CP2K/MACE-written extxyz.
# ---------------------------------------------------------------------------


def iter_raw_blocks(path: Path) -> list[str]:
    lines = Path(path).read_text().splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        block = "\n".join(lines[i : i + 2 + n_atoms])
        blocks.append(block)
        i += 2 + n_atoms
    return blocks


def write_raw_blocks(path: Path, blocks: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(blocks) + ("\n" if blocks else ""))


def extract_by_indices(path: Path, indices: list[int], out_path: Path) -> Path:
    blocks = iter_raw_blocks(path)
    write_raw_blocks(out_path, [blocks[i] for i in indices])
    return Path(out_path)


def train_test_split_raw(
    path: Path, test_fraction: float, train_out: Path, test_out: Path, seed: int = 0
) -> tuple[Path, Path]:
    import random

    blocks = iter_raw_blocks(path)
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)
    n_test = max(1, int(len(order) * test_fraction)) if blocks else 0
    test_idx, train_idx = set(order[:n_test]), set(order[n_test:])
    write_raw_blocks(train_out, [b for i, b in enumerate(blocks) if i in train_idx])
    write_raw_blocks(test_out, [b for i, b in enumerate(blocks) if i in test_idx])
    return Path(train_out), Path(test_out)
