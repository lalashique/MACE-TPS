"""
util/mlp_helper.py — Per-frame uncertainty metrics for committee MLP models.

Provides ``process_frame``, which extracts energy variance, force variance,
and local force variance (around a set of key atoms) from a single ASE Atoms
object that has already been evaluated by ``MACE_MLP.evaluate_committee``.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def process_frame(
    atoms: Atoms,
    carbon_indices: list[int] | np.ndarray,
    cutoff: float,
) -> tuple[float, float, float, float]:
    """
    Extract uncertainty metrics from a committee-evaluated frame.

    The frame must have been processed by ``MACE_MLP.evaluate_committee``,
    which stores the following keys:

    - ``atoms.info["energy_variance"]``    — scalar energy variance (eV²)
    - ``atoms.info["force_variance"]``     — mean force variance (eV²/Å²)
    - ``atoms.arrays["force_var_per_atom"]`` — per-atom force variance (eV²/Å²)

    Parameters
    ----------
    atoms : ase.Atoms
        A single frame with committee statistics already populated.
    carbon_indices : array-like of int
        Indices of the "key" atoms that define the local region of interest
        (not necessarily carbon — the name is a historical artefact).
    cutoff : float
        Distance cutoff in Å.  All atoms within this distance of any key atom
        are included in the local statistics.

    Returns
    -------
    energy_var : float
        Global energy variance across committee models (eV²).
    force_var : float
        Global mean force variance across committee models (eV²/Å²).
    local_mean : float
        Mean per-atom force variance within the local region (eV²/Å²),
        or ``np.nan`` if no atoms fall within the cutoff.
    local_max : float
        Maximum per-atom force variance within the local region (eV²/Å²),
        or ``np.nan`` if no atoms fall within the cutoff.
    """
    energy_var = atoms.info.get("energy_variance")
    force_var  = atoms.info.get("force_variance")
    atom_var   = atoms.arrays.get("force_var_per_atom")

    if energy_var is None:
        raise KeyError("'energy_variance' not found in atoms.info. "
                       "Run MACE_MLP.evaluate_committee first.")
    if force_var is None:
        raise KeyError("'force_variance' not found in atoms.info.")
    if atom_var is None:
        raise KeyError("'force_var_per_atom' not found in atoms.arrays.")

    # Vectorised local distance computation
    key_indices = np.asarray(carbon_indices)
    positions   = atoms.positions                        # (N, 3)
    key_pos     = positions[key_indices]                 # (K, 3)

    # Distance of every atom to the nearest key atom
    # Shape: (K, N) → min over K → (N,)
    diff  = positions[None, :, :] - key_pos[:, None, :]  # (K, N, 3)
    dists = np.linalg.norm(diff, axis=-1).min(axis=0)    # (N,)

    local_idx = np.where(dists < cutoff)[0]

    if len(local_idx) > 0:
        local_var = atom_var[local_idx]
        local_mean = float(np.mean(local_var))
        local_max  = float(np.max(local_var))
    else:
        local_mean = np.nan
        local_max  = np.nan

    return float(energy_var), float(force_var), local_mean, local_max
