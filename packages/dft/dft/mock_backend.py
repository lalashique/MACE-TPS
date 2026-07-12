"""Instant, zero-dependency stand-in for a real DFT engine.

Used for the fast end-to-end smoke test - this does NOT compute anything
physically meaningful. It exists purely to exercise the workflow plumbing
(file formats, generation loop, gates) without paying for CP2K.

Energy/forces come from a deterministic toy potential (a harmonic well
around the origin, per atom) so that:
  - runs are reproducible (same structures -> same "labels" every time)
  - downstream mock MLP committees can add a small seed-dependent bias to
    the same toy potential, giving nonzero but bounded committee variance
    for the uncertainty-driven frame selection to actually do something.
"""

from __future__ import annotations

from pathlib import Path

from core.interfaces import DFTEngine
from core.xyz_io import Frame, read_frames

_K = 0.05  # toy harmonic spring constant
_BOX = 20.0  # dummy non-periodic box, large enough to not matter physically


def toy_energy_and_forces(frame: Frame) -> tuple[float, list[tuple[float, float, float]]]:
    energy = 0.0
    forces = []
    for x, y, z in frame.positions:
        energy += 0.5 * _K * (x * x + y * y + z * z)
        forces.append((-_K * x, -_K * y, -_K * z))
    return energy, forces


def _write_real_extxyz(
    path: Path,
    frames: list[Frame],
    energies: list[float],
    forces_per_frame: list[list[tuple[float, float, float]]],
    energy_key: str,
    forces_key: str,
) -> None:
    """Write genuine ASE-compatible extended XYZ, not the minimal format in
    core.xyz_io. This matters because this file is meant to be consumed by
    the *real* MLP backend (mace.cli.run_train via ase.io.read) - without a
    proper Properties= header and a matching energy/forces key in the
    comment line, ase's extxyz reader can't find the labels at all and
    training fails with "None of 'energy_cp2k', 'forces_cp2k', ... found".
    core.xyz_io's Frame format (just "energy=") was never meant to satisfy
    that - it's only for the mock backends talking to each other.
    """
    lines = []
    for frame, energy, forces in zip(frames, energies, forces_per_frame):
        lines.append(str(frame.n_atoms))
        lattice = f'{_BOX} 0.0 0.0 0.0 {_BOX} 0.0 0.0 0.0 {_BOX}'
        properties = f"species:S:1:pos:R:3:{forces_key}:R:3"
        lines.append(
            f'Lattice="{lattice}" Properties={properties} {energy_key}={energy} pbc="F F F"'
        )
        for (x, y, z), (fx, fy, fz), symbol in zip(frame.positions, forces, frame.symbols):
            lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


class MockDFTEngine(DFTEngine):
    def __init__(self, energy_key: str = "energy_cp2k", forces_key: str = "forces_cp2k"):
        self.energy_key = energy_key
        self.forces_key = forces_key

    def label(self, structures: Path, workdir: Path) -> Path:
        frames = read_frames(structures)
        energies, forces_per_frame = [], []
        for frame in frames:
            energy, forces = toy_energy_and_forces(frame)
            energies.append(energy)
            forces_per_frame.append(forces)

        out_path = Path(workdir) / "labeled.xyz"
        _write_real_extxyz(out_path, frames, energies, forces_per_frame, self.energy_key, self.forces_key)
        return out_path
