"""
util/traj_util.py — Trajectory manipulation and frame-selection utilities.

Provides:
- ``train_test_split``  — shuffle and split an XYZ dataset.
- ``select_frames``     — pick high-uncertainty frames from a trajectory.
- ``split_times``       — split a trajectory into N roughly equal segments.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase.io import read, write
from loguru import logger


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def train_test_split(
    xyz: str | Path,
    test_fraction: float = 0.25,
    seed: int | None = None,
) -> tuple[list, list]:
    """
    Shuffle and split an extended XYZ file into train and test sets.

    Writes ``train.xyz`` and ``test.xyz`` into the same directory as ``xyz``.

    Parameters
    ----------
    xyz : str or Path
        Path to the source XYZ file.
    test_fraction : float
        Fraction of frames to reserve for the test set (default 0.25).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    train_atoms, test_atoms : list[ase.Atoms]
    """
    xyz = Path(xyz)
    if not xyz.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz}")

    atoms_list = read(str(xyz), ":")
    total      = len(atoms_list)
    rng        = np.random.default_rng(seed)
    indices    = rng.permutation(total)

    cut         = int((1 - test_fraction) * total)
    train_atoms = [atoms_list[i] for i in indices[:cut]]
    test_atoms  = [atoms_list[i] for i in indices[cut:]]

    folder = xyz.parent
    write(str(folder / "train.xyz"), train_atoms)
    write(str(folder / "test.xyz"),  test_atoms)
    logger.info(f"Split {total} frames → {len(train_atoms)} train, {len(test_atoms)} test.")
    return train_atoms, test_atoms


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames(
    xyz_path: str | Path,
    tps_params: dict,
    out_file: str | Path = "selected.xyz",
) -> list:
    """
    Select high-uncertainty frames from a committee-evaluated trajectory.

    Selection criteria (all taken from ``tps_params``):

    - Worst ``num_select`` frames by global energy variance.
    - Worst ``num_select`` frames by global force variance.
    - Worst ``num_select`` frames by local mean/max per-atom force variance
      within windows centred on each index in ``selection_indices``.

    The final pool is deduplicated and randomly down-sampled to ``num_select``
    if larger.

    Parameters
    ----------
    xyz_path : str or Path
        Extended XYZ file with committee statistics in ``atoms.info`` /
        ``atoms.arrays``.
    tps_params : dict
        Must contain:

        - ``"num_select"``        (int) — target number of frames to output.
        - ``"selection_indices"`` (list[int]) — key-atom indices for local stats.
        - ``"local_range"``       (int) — half-width of the frame window.

    out_file : str or Path
        Output XYZ file path.

    Returns
    -------
    list[ase.Atoms]
        Selected frames (also written to ``out_file``).
    """
    xyz_path = Path(xyz_path)
    out_file = Path(out_file)

    frames = read(str(xyz_path), ":")
    n      = len(frames)

    selection_indices = tps_params.get("selection_indices")
    local_range       = tps_params.get("local_range", 50)
    num_select        = tps_params.get("num_select", 200)

    # Extract metrics
    try:
        energy_var = np.array([f.info["energy_variance"]    for f in frames])
        force_var  = np.array([f.info["force_variance"]     for f in frames])
        fpa        = np.array([f.arrays["force_var_per_atom"] for f in frames])
    except KeyError as exc:
        raise KeyError(
            f"Required key {exc} missing from trajectory frames. "
            "Ensure committee evaluation was run before frame selection."
        ) from exc

    fpa_mean = fpa.mean(axis=1)
    fpa_max  = fpa.max(axis=1)

    collected: list[np.ndarray] = []

    # Global energy variance
    collected.append(np.argsort(energy_var)[-num_select:])
    # Global force variance
    collected.append(np.argsort(force_var)[-num_select:])

    # Local windows around selection_indices
    if selection_indices:
        for metric in (fpa_mean, fpa_max):
            picks = []
            for idx in selection_indices:
                lo = max(idx - local_range, 0)
                hi = min(idx + local_range + 1, n)
                window = np.arange(lo, hi)
                worst  = window[np.argsort(metric[window])[-num_select:]]
                picks.append(worst)
            collected.append(np.unique(np.concatenate(picks)))

    final_ids = np.unique(np.concatenate(collected))

    # Rank by combined severity and down-sample if needed
    severity = energy_var[final_ids] + force_var[final_ids] + fpa_mean[final_ids] + fpa_max[final_ids]
    if len(final_ids) > num_select:
        top_k     = np.argsort(severity)[-num_select:]
        final_ids = final_ids[top_k]

    selected = [frames[i] for i in final_ids]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_file), selected)
    logger.info(f"Selected {len(selected)} frames → {out_file}")
    return selected


# ---------------------------------------------------------------------------
# Trajectory splitting
# ---------------------------------------------------------------------------

def split_times(xyz: str | Path, split: int = 4, seed: int | None = None) -> list[Path]:
    """
    Shuffle a trajectory and split it into ``split`` roughly equal segments.

    Writes ``<stem>_part1.xyz``, ``<stem>_part2.xyz``, … alongside the source file.

    Parameters
    ----------
    xyz : str or Path
    split : int
        Number of segments.
    seed : int, optional
        Random seed.

    Returns
    -------
    list[Path]
        Paths to the written segment files.
    """
    xyz = Path(xyz)
    if not xyz.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz}")

    atoms_list = read(str(xyz), ":")
    rng        = np.random.default_rng(seed)
    rng.shuffle(atoms_list)

    total  = len(atoms_list)
    size   = total // split
    paths_out: list[Path] = []

    for i in range(split):
        start   = i * size
        end     = (i + 1) * size if i < split - 1 else total
        segment = atoms_list[start:end]
        out     = xyz.parent / f"{xyz.stem}_part{i + 1}{xyz.suffix}"
        write(str(out), segment)
        paths_out.append(out)
        logger.info(f"Segment {i + 1}/{split}: {len(segment)} frames → {out}")

    return paths_out
