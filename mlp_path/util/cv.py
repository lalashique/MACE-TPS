"""
util/cv.py — Collective variable (CV) definitions and state construction.

The ``CV`` class reads a CV block from the JSON config, builds OPS
``CoordinateFunctionCV`` objects, plots each CV along the initial trajectory,
and constructs the bound/unbound metastable states used by the TPS/TIS sampler.

Config schema (``"CV"`` block)
-------------------------------
    "CV": {
        "my_cv_name": {
            "type": "<method_name>",      # static method on CV class
            "params": {
                "group_A_atoms": "name CA",   # MDAnalysis selection string
                ...
            },
            "bound":   [min, max],        # optional
            "unbound": [min, max]         # optional
        }
    }

Any string-valued parameter is interpreted as an MDAnalysis atom-selection
expression and resolved to an array of integer atom indices.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import mdtraj
from loguru import logger

from .parser import read_config


class CV:
    """
    Build collective variables and metastable state volumes from a JSON config.

    Parameters
    ----------
    config : dict
        Parsed JSON config (already loaded).
    traj_path : str
        Path to the initial trajectory XYZ file.

    Attributes
    ----------
    states : tuple[paths.Volume, paths.Volume]
        ``(bound_state, unbound_state)`` as OPS ``CVDefinedVolume`` objects.
    descriptor_transform : callable
        Maps an OPS snapshot to a 1-D numpy array of CV values.
    """

    def __init__(self, config: dict | str | Path, traj_path: str = "p1.xyz") -> None:
        # Accept either an already-parsed dict or a path to a JSON config file,
        # so that both old call-sites (CV(config_path, traj)) and new ones
        # (CV(config_dict, traj)) continue to work.
        if isinstance(config, (str, Path)):
            config = read_config(config)
        self.config = config
        cv_config = config.get("CV", {})

        logger.debug("Loading topology and initial trajectory for CV setup.")
        top_file = config.get("top_file", "top.pdb")
        mdt_top  = mdtraj.load(top_file)
        traj     = mdtraj.load(traj_path, top=mdt_top)
        traj.unitcell_vectors = np.array(list(mdt_top.unitcell_vectors) * len(traj.xyz))
        init_traj = ops_openmm.trajectory_from_mdtraj(
            traj, velocities=[np.zeros_like(f) for f in traj.xyz]
        )

        logger.debug(f"Defining CVs: {list(cv_config.keys())}")
        cvs: list[paths.CoordinateFunctionCV] = []

        for cv_name, cv_spec in cv_config.items():
            func = getattr(self, cv_spec["type"])
            params = self._resolve_atom_selections(cv_spec["params"], traj_path)

            cv_obj = paths.CoordinateFunctionCV(
                name=cv_name, f=func, **params
            ).with_diskcache()
            cvs.append(cv_obj)

            # Plot CV along initial trajectory
            fig, ax = plt.subplots()
            ax.plot(cv_obj(init_traj))
            ax.set_xlabel("Frame")
            ax.set_ylabel(cv_name)
            fig.savefig(f"{cv_name}_initial_trajectory.png", dpi=150)
            plt.close(fig)
            logger.debug(f"Saved CV plot: {cv_name}_initial_trajectory.png")

        # Build state volumes
        bound_vols:   list[paths.Volume] = []
        unbound_vols: list[paths.Volume] = []

        for cv_obj in cvs:
            spec = cv_config[cv_obj.name]
            if "bound" in spec:
                lo, hi = spec["bound"]
                bound_vols.append(
                    paths.CVDefinedVolume(cv_obj, lambda_min=lo, lambda_max=hi)
                    .named(f"bound_{cv_obj.name}")
                )
            if "unbound" in spec:
                lo, hi = spec["unbound"]
                unbound_vols.append(
                    paths.CVDefinedVolume(cv_obj, lambda_min=lo, lambda_max=hi)
                    .named(f"unbound_{cv_obj.name}")
                )

        if not bound_vols or not unbound_vols:
            raise ValueError(
                "At least one CV must define both 'bound' and 'unbound' ranges."
            )

        # Combine multiple bounds/unbounds with logical AND
        bound_state   = bound_vols[0]
        unbound_state = unbound_vols[0]
        for vol in bound_vols[1:]:
            bound_state = bound_state & vol
        for vol in unbound_vols[1:]:
            unbound_state = unbound_state & vol

        self.states = (bound_state, unbound_state)

        # Descriptor transform: snapshot → 1-D array of CV values
        self.descriptor_transform = paths.FunctionCV(
            "descriptor_transform",
            lambda snapshot, _cvs=cvs: np.array([c(snapshot) for c in _cvs]),
            cv_wrap_numpy_array=True,
        ).with_diskcache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_atom_selections(params: dict, traj_path: str) -> dict:
        """
        Replace MDAnalysis selection strings with integer index arrays.
        Non-string parameters are passed through unchanged.
        """
        resolved = {}
        universe = None  # lazy init
        for key, val in params.items():
            if isinstance(val, str):
                if universe is None:
                    universe = mda.Universe(traj_path)
                resolved[key] = universe.select_atoms(val).indices
            else:
                resolved[key] = val
        return resolved

    # ------------------------------------------------------------------
    # CV functions (static methods)
    # ------------------------------------------------------------------

    @staticmethod
    def bond(
        snapshot,
        group_A_atoms: np.ndarray,
        group_B_atoms: np.ndarray,
        type: int = 0,
        axis: int = -1,
    ) -> float:
        """
        Distance-based CV between two groups of atoms.

        Parameters
        ----------
        type : {0, -1, 1}
            0  → distance between group COMs (or projection on ``axis``).
            -1 → minimum pairwise distance.
            1  → maximum pairwise distance.
        axis : int
            Cartesian axis index (0=x, 1=y, 2=z) for projection, or -1 for norm.
        """
        A = np.atleast_2d(snapshot.xyz[group_A_atoms, :])
        B = np.atleast_2d(snapshot.xyz[group_B_atoms, :])

        if type == 0:
            diff = A.mean(axis=0) - B.mean(axis=0)
            return float(diff[axis] if axis >= 0 else np.linalg.norm(diff))

        dists = distance_array(A.reshape(-1, 3), B.reshape(-1, 3))
        if type == -1:
            return float(np.min(dists[:, axis]) if axis >= 0 else np.min(dists))
        if type == 1:
            return float(np.max(dists[:, axis]) if axis >= 0 else np.max(dists))
        raise ValueError(f"Unknown bond type: {type}")

    @staticmethod
    def bondfunction(
        snapshot,
        group_A_atoms: np.ndarray,
        group_B_atoms: np.ndarray,
        group_C_atoms: np.ndarray,
        group_D_atoms: np.ndarray,
        coeff: float = -1,
    ) -> float:
        """Linear combination of two inter-group distances: |AB| + coeff * |CD|."""
        ab = snapshot.xyz[group_A_atoms, :] - snapshot.xyz[group_B_atoms, :]
        cd = snapshot.xyz[group_C_atoms, :] - snapshot.xyz[group_D_atoms, :]
        return float(np.linalg.norm(ab) + coeff * np.linalg.norm(cd))

    @staticmethod
    def CN(
        snapshot,
        group_A_atoms: np.ndarray,
        group_B_atoms: np.ndarray,
        d0: float = 1.25,
        n: int = 12,
        m: int = 24,
    ) -> float:
        """
        Coordination number using the rational switching function.

        s_ij = (1 - (r/d0)^n) / (1 - (r/d0)^m)

        Parameters
        ----------
        d0 : float
            Reference distance in Å (converted internally to nm).
        n, m : int
            Exponents of the switching function.
        """
        d0_nm = d0 / 10  # Å → nm
        o1 = snapshot.xyz[group_A_atoms, :]
        h  = snapshot.xyz[group_B_atoms, :]
        dist = distance_array(o1, h)
        s_ij = (1 - (dist / d0_nm) ** n) / (1 - (dist / d0_nm) ** m)
        return float(np.sum(s_ij))

    @staticmethod
    def bond_count(
        snapshot,
        group_A_atoms: np.ndarray,
        group_B_atoms: np.ndarray,
        value: float = 1.5,
        rounding: int = 2,
    ) -> int:
        """Count atom pairs whose distance equals ``value`` Å (after rounding)."""
        A = np.atleast_2d(snapshot.xyz[group_A_atoms, :]).reshape(-1, 3)
        B = np.atleast_2d(snapshot.xyz[group_B_atoms, :]).reshape(-1, 3)
        dists = distance_array(A, B)
        return int(np.sum(np.round(dists, rounding) == value))

    @staticmethod
    def bond_plane_angle(
        snapshot,
        group_A_atoms: np.ndarray,
        group_B_atoms: np.ndarray,
        axis: int = 2,
    ) -> float:
        """
        Angle (degrees) between the A→B COM vector and the plane perpendicular
        to ``axis``.  Returns values in [-90, 90].
        """
        A = np.atleast_2d(snapshot.xyz[group_A_atoms, :]).mean(axis=0)
        B = np.atleast_2d(snapshot.xyz[group_B_atoms, :]).mean(axis=0)
        ab = A - B
        return float(np.degrees(np.arccos(ab[axis] / np.linalg.norm(ab))) - 90)

    @staticmethod
    def distance_to_OH(
        snapshot,
        O_indices: np.ndarray,
        H_indices: np.ndarray,
        atom: int,
        bond_cutoff: float = 1.3,
        box: np.ndarray | None = None,
    ) -> float:
        """
        Distance from ``atom`` to the other hydroxide oxygen in the simulation.

        Returns ``np.inf`` if fewer than two OH groups are detected.

        Parameters
        ----------
        O_indices, H_indices : array of int
            Indices of oxygen and hydrogen atoms in the system.
        atom : int
            Index of the reference oxygen atom.
        bond_cutoff : float
            O–H bond cutoff in Å.
        box : array of shape (6,), optional
            Periodic box dimensions [Lx, Ly, Lz, α, β, γ] in Å.
        """
        from MDAnalysis.lib.distances import capped_distance

        pairs, _ = capped_distance(
            snapshot.xyz[0, O_indices, :],
            snapshot.xyz[0, H_indices, :],
            max_cutoff=bond_cutoff / 10,
            box=box,
        )
        if len(pairs) == 0:
            return np.inf

        unique_O, h_counts = np.unique(pairs[:, 0], return_counts=True)
        oh_indices = unique_O[h_counts == 1]  # oxygens bonded to exactly one H

        if len(oh_indices) < 2:
            return np.inf

        oh_oxygens = O_indices[oh_indices]
        other_O    = snapshot.xyz[0, oh_oxygens[oh_oxygens != atom], :]
        if len(other_O) == 0:
            return np.inf

        return float(np.linalg.norm(snapshot.xyz[0, atom] - other_O[0]))
