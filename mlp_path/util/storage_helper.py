"""
util/storage_helper.py — Extract trajectories from OPS storage files.

Provides ``save_from_storage``, which iterates over every Monte Carlo step in
an OpenPathSampling ``Storage`` file and saves accepted and rejected
trajectories as XYZ files, along with a ``shooting_traj.xyz`` that marks the
shooting-point snapshot for each step.

CLI usage
---------
    python -m util.storage_helper \
        --storage tps_storage.db \
        --config  tps.json \
        --out     traj/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from loguru import logger

import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all, Storage

paths = monkey_patch_all(paths)


# ---------------------------------------------------------------------------
# XYZ serialisation helpers
# ---------------------------------------------------------------------------

def _traj_to_ase(trajectory, topology) -> list[Atoms]:
    """Convert an OPS trajectory to a list of ASE Atoms objects."""
    symbols = [
        atom.element.symbol if atom.element else atom.name[0]
        for atom in topology.atoms
    ]
    atoms_list = []
    for snap in trajectory:
        atoms_list.append(
            Atoms(symbols=symbols, positions=snap.xyz * 10)  # nm → Å
        )
    return atoms_list


def _save_traj(
    trajectory,
    file_name: str | Path,
    topology=None,
    fmt: str = "mdtraj",
) -> None:
    """
    Save an OPS trajectory to an XYZ file.

    Parameters
    ----------
    fmt : {"mdtraj", "ase"}
        ``"mdtraj"`` uses the built-in mdtraj serialiser (faster).
        ``"ase"`` converts to ASE Atoms and writes extxyz.
    """
    file_name = Path(file_name)
    file_name.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "mdtraj":
        trajectory.to_mdtraj().save(str(file_name))
    elif fmt == "ase":
        if topology is None:
            raise ValueError("topology must be provided for ASE format.")
        write(str(file_name), _traj_to_ase(trajectory, topology))
    else:
        raise ValueError(f"Unsupported format: '{fmt}'")


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def save_from_storage(
    storage_path: str | Path,
    config: str | Path,
    out_folder: str | Path,
) -> None:
    """
    Extract accepted and rejected trajectories from an OPS storage file.

    Resumes from where a previous run left off: any step whose output file
    already exists in ``out_folder`` is skipped.

    Output files per step ``i``:
    - ``accepted_{i}.xyz``       — accepted transition path
    - ``rejected_max_{i}.xyz``   — rejected (max-length reached)
    - ``rejected_same_{i}.xyz``  — rejected (same state)
    - ``shooting_traj.xyz``      — all shooting-point snapshots

    Statistics (acceptance rate, rejection breakdown) are logged at INFO level.

    Parameters
    ----------
    storage_path : str or Path
        OPS ``.db`` storage file produced by the TPS run.
    config : str or Path
        JSON config file (same one passed to ``TPS_setup``).
    out_folder : str or Path
        Directory to write trajectory files into.
    """
    import tps as tps_module

    storage_path = Path(storage_path)
    out_folder   = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    setup = tps_module.TPS_setup(config)
    paths.PathMover.engine = setup.md_engine

    logger.info(f"Loading OPS storage: {storage_path}")
    storage = Storage(storage_path, "r")
    storage.engine = None

    n_steps = len(storage.steps)
    logger.info(f"Total steps: {n_steps},  trajectories: {len(storage.trajectories)}")

    # Determine resume index
    existing = set(os.listdir(out_folder))
    resume_from = 0
    for i in range(1_000_000):
        if any(
            f"{kind}_{i}.xyz" in existing
            for kind in ("accepted", "rejected_max", "rejected_same")
        ):
            resume_from = i + 1
        else:
            break
    logger.info(f"Resuming from step index {resume_from}.")

    save_rejected = setup.config.get("save_rejected", False)
    shooting_snaps: list[Atoms] = []
    accepted = rejected_same = rejected_max = 0

    for idx in range(resume_from, n_steps):
        step = storage.steps[idx]
        traj = storage.trajectories[idx]

        # --- Shooting snapshot ---
        try:
            snap    = step.change.canonical.details.shooting_snapshot
            top_mdt = snap.topology.mdtraj
            symbols = [
                a.element.symbol if a.element else a.name[0] for a in top_mdt.atoms
            ]
            snap_atoms = Atoms(symbols=symbols, positions=snap.xyz * 10)
            snap_atoms.info["shooting_state"] = (
                "accepted" if step.change.accepted else
                ("rejected_max_length" if len(traj) >= setup.n_frames_max
                 else "rejected_same_state")
            )
            shooting_snaps.append(snap_atoms)
            topology = top_mdt
        except Exception as exc:
            logger.warning(f"Step {idx}: could not extract shooting snapshot — {exc}")
            topology = None

        # --- Save trajectory ---
        if step.change.accepted:
            _save_traj(traj, out_folder / f"accepted_{idx}.xyz")
            accepted += 1
        elif len(traj) >= setup.n_frames_max:
            if save_rejected:
                _save_traj(traj, out_folder / f"rejected_max_{idx}.xyz",
                           topology=topology, fmt="ase")
            rejected_max += 1
        else:
            if save_rejected:
                _save_traj(traj, out_folder / f"rejected_same_{idx}.xyz",
                           topology=topology, fmt="ase")
            rejected_same += 1

    # Write shooting-point trajectory
    if shooting_snaps:
        write(str(out_folder / "shooting_traj.xyz"), shooting_snaps)

    total = accepted + rejected_same + rejected_max
    acceptance_rate = round(accepted / total, 3) if total > 0 else 0.0
    logger.info(
        f"Done. Accepted: {accepted}, "
        f"Rejected (same state): {rejected_same}, "
        f"Rejected (max length): {rejected_max}. "
        f"Acceptance rate: {acceptance_rate:.1%}"
    )
    storage.close()


# ---------------------------------------------------------------------------
# Utility: snapshot to XYZ from PDB topology
# ---------------------------------------------------------------------------

def save_xyz_from_pdb(snapshot, pdb_file: str, fname: str) -> None:
    """
    Write a single OPS snapshot to a plain XYZ file using a PDB for atom symbols.

    Parameters
    ----------
    snapshot : OPS snapshot
    pdb_file : str
        Path to a PDB file providing atom element information.
    fname : str
        Output XYZ file path.
    """
    from openmm.app import PDBFile

    pdb   = PDBFile(pdb_file)
    atoms = list(pdb.topology.atoms())
    pos   = snapshot.xyz * 10  # nm → Å

    with open(fname, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom, p in zip(atoms, pos):
            elem = atom.element.symbol if atom.element else atom.name[0]
            f.write(f"{elem} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract trajectories from an OPS storage file.")
    p.add_argument("--storage", required=True, help="Path to the OPS .db storage file.")
    p.add_argument("--config",  required=True, help="Path to the TPS JSON config file.")
    p.add_argument("--out",     required=True, help="Output directory for trajectory files.")
    args = p.parse_args()

    save_from_storage(args.storage, args.config, args.out)
