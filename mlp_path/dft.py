"""
dft.py — DFT engine interface for the mlpath active-learning workflow.

Provides a unified abstract base (DFTBase) and a concrete CP2K implementation
(CP2KDFT). New engines can be added by subclassing DFTBase and registering
them with DFTFactory.

Typical usage
-------------
    from dft import DFTFactory

    engine = DFTFactory.create(
        engine="cp2k",
        settings={"command": "mpirun -np 20 cp2k.popt"},
        input_files=["cp2k.inp", "force_eval.inp"],
        reftraj=True,
        run_mode="prepare",
        workdir="dft_output",
    )
    engine.run(["coord_0.xyz", "coord_1.xyz"])
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from tqdm import tqdm

logger.add("dft_debug.log", level="INFO")

# Unit conversion constants
_HA_TO_EV: float = 27.211        # Hartree → eV
_FORCE_AU_TO_EV_ANG: float = 51.422  # a.u. force → eV/Å


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DFTBase(ABC):
    """
    Abstract base class for all DFT engine wrappers.

    Subclasses must set ``supports_reftraj`` and implement the three abstract
    methods: ``_prepare_structure``, ``_run_calculation``, ``_parse_outputs``.

    Parameters
    ----------
    engine : str
        Engine identifier (e.g. ``"cp2k"``).
    settings : dict
        Engine-specific settings dict (e.g. ``{"command": "mpirun -np 20 cp2k.popt"}``).
    input_files : list[str]
        Template input file paths to copy into each structure directory.
    reftraj : bool
        Enable multi-structure reftraj mode.  Raises if unsupported.
    run_mode : {"prepare", "run"}
        ``"prepare"`` writes input files only; ``"run"`` also executes.
    workdir : str
        Root directory for all DFT inputs/outputs.
    """

    supports_reftraj: bool = False

    def __init__(
        self,
        engine: str,
        settings: dict,
        input_files: list[str],
        reftraj: bool = False,
        run_mode: Literal["prepare", "run"] = "prepare",
        workdir: str = "output",
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.reftraj = reftraj
        self.run_mode = run_mode
        self.workdir = Path(workdir)
        self.input_files = [Path(f) for f in input_files]
        self.completed = False
        self.results = None
        self.state: dict = {"engine": self.engine, "run_mode": self.run_mode}
        self._validate_capabilities()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, structures: list[str]) -> None:
        """
        Prepare inputs and (optionally) execute the DFT calculation.

        Parameters
        ----------
        structures : list[str]
            Paths to coordinate files.
        """
        logger.debug(f"[{self.engine.upper()}] run() — {len(structures)} structure(s).")
        self._prepare_structure(structures)

        if self.run_mode == "prepare":
            logger.info(f"[{self.engine.upper()}] Inputs prepared (mode={self._mode_name()}).")
            self._save_state()
            return

        self._run_calculation(structures)
        self._parse_outputs()
        self.completed = True
        self._save_state()
        logger.info(f"[{self.engine.upper()}] Run complete (mode={self._mode_name()}).")

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare_structure(self, structures: list[str]) -> None: ...

    @abstractmethod
    def _run_calculation(self, structures: list[str]) -> None: ...

    @abstractmethod
    def _parse_outputs(self) -> None: ...

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _mode_name(self) -> str:
        return "reftraj" if self.reftraj else "single"

    def _save_state(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        with open(self.workdir / "state.json", "w") as f:
            json.dump(self.state, f, indent=2)

    def _load_state(self) -> dict:
        state_file = self.workdir / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")
        with open(state_file) as f:
            return json.load(f)

    def _validate_capabilities(self) -> None:
        if self.reftraj and not self.supports_reftraj:
            raise ValueError(
                f"[{self.engine.upper()}] reftraj mode is not supported. "
                "Set reftraj=False or use CP2K."
            )

    @staticmethod
    def _copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# CP2K
# ---------------------------------------------------------------------------

class CP2KDFT(DFTBase):
    """
    CP2K DFT engine wrapper.

    Supports single-structure and reftraj (multi-structure batch) modes.
    In reftraj mode each structure gets its own subdirectory containing all
    required input files, a ``coord.xyz``, and an empty ``PAUSE`` sentinel.
    After CP2K finishes and the user removes PAUSE, ``_parse_outputs`` reads
    energies/forces/positions and writes an ``extdata.xyz`` per structure.
    """

    supports_reftraj = True

    def __init__(
        self,
        settings: dict,
        input_files: list[str],
        reftraj: bool = False,
        run_mode: str = "run",
        workdir: str = "dft_cp2k",
    ) -> None:
        super().__init__("cp2k", settings, input_files, reftraj, run_mode, workdir)

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def _prepare_structure(self, structures: list[str]) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self.reftraj:
            self._write_reftraj_inputs(structures)
        else:
            self._write_single_inputs(structures)

    def _write_reftraj_inputs(self, structures: list[str]) -> None:
        for i, struct in enumerate(structures):
            struct_path = Path(struct).resolve()
            struct_dir = self.workdir / f"structure_{i}"
            struct_dir.mkdir(parents=True, exist_ok=True)

            for tmpl in self.input_files:
                self._copy(tmpl, struct_dir / tmpl.name)

            self._copy(struct_path, struct_dir / "coord.xyz")
            (struct_dir / "PAUSE").touch()

            self.state[f"structure_{i}"] = {"status": 0, "file": str(struct_path)}
            logger.debug(f"[CP2K] Prepared structure {i} → {struct_dir}")
            logger.info(
                f"[CP2K] Run command for structure {i}:\n"
                f"  cd {struct_dir} && "
                f"{self.settings['command']} -i {self.input_files[0].name} -o cp2k.out"
            )

    def _write_single_inputs(self, structures: list[str]) -> None:
        if not structures:
            raise ValueError("[CP2K] No structures provided.")
        struct_path = Path(structures[0]).resolve()
        for tmpl in self.input_files:
            self._copy(tmpl, self.workdir / tmpl.name)
        self._copy(struct_path, self.workdir / "coord.xyz")
        (self.workdir / "PAUSE").touch()
        self.state["structure_0"] = {"status": 0, "file": str(struct_path)}
        logger.info(
            f"[CP2K] Run command:\n"
            f"  cd {self.workdir} && "
            f"{self.settings['command']} -i {self.input_files[0].name} -o cp2k.out"
        )

    # ------------------------------------------------------------------
    # Run (external HPC job; placeholder)
    # ------------------------------------------------------------------

    def _run_calculation(self, structures: list[str]) -> None:
        logger.info("[CP2K] External HPC submission required — skipping direct execution.")

    # ------------------------------------------------------------------
    # Parse outputs
    # ------------------------------------------------------------------

    def _parse_outputs(self) -> None:
        """
        For each ``structure_N`` in state, read CP2K output files and write
        an ``extdata.xyz`` containing positions, forces (eV/Å), and energies (eV).
        Only converged steps are included.
        """
        import MDAnalysis as mda
        from ase import Atoms
        from ase.io import write
        import cpoker.iparser as ip
        import cpoker.genfun as gf

        for struct_key in self.state:
            if not struct_key.startswith("structure_"):
                continue

            struct_dir = self.workdir / struct_key
            logger.debug(f"[CP2K] Parsing {struct_key}")

            u_pos   = mda.Universe(str(struct_dir / "cp2k-pos-1.xyz"))
            u_force = mda.Universe(str(struct_dir / "cp2k-frc-1.xyz"))

            # Energies: prefer .ener file, fall back to XYZ comment field
            ener_file = struct_dir / "cp2k-1.ener"
            try:
                energies = np.loadtxt(str(ener_file))[:, 4]
            except Exception as exc:
                logger.warning(f"[CP2K] Cannot read {ener_file}: {exc}. Falling back to XYZ.")
                raw = gf.run_command(
                    f"grep i {struct_dir}/cp2k-pos-1.xyz | awk '{{print $NF}}'", out=True
                )
                energies = np.array(raw.strip().split("\n"), dtype=float)

            n_frames = len(u_pos.trajectory)
            if not (n_frames == len(u_force.trajectory) == len(energies)):
                raise ValueError(
                    f"[CP2K] {struct_key}: frame count mismatch — "
                    f"pos={n_frames}, frc={len(u_force.trajectory)}, ener={len(energies)}"
                )

            # Cell parameters from CP2K input
            cell = [float(x) for x in ip.iparser(str(struct_dir)).val_dict["ABC"].split()]

            # Convergence mask
            conv_raw = gf.run_command(f"grep converge {struct_dir}/cp2k.out", out=True)
            conv_lines = [l for l in conv_raw.strip().split("\n") if l]
            converged = np.array(["NOT" not in l for l in conv_lines])[: len(energies)]
            logger.info(
                f"[CP2K] {struct_key}: {converged.sum()}/{len(converged)} converged steps."
            )

            atoms_list = []
            for ts_pos, ts_force, energy in tqdm(
                zip(
                    u_pos.trajectory[converged],
                    u_force.trajectory[converged],
                    energies[converged],
                ),
                total=int(converged.sum()),
                desc=f"Parsing {struct_key}",
            ):
                forces = ts_force.positions * _FORCE_AU_TO_EV_ANG
                forces[np.isclose(forces, 0.0)] = np.nan  # mark unconverged forces

                atoms = Atoms(
                    symbols=u_pos.atoms.elements,
                    positions=ts_pos.positions,
                    cell=cell,
                    pbc=True,
                )
                atoms.info["energy_cp2k"]   = float(energy) * _HA_TO_EV
                atoms.arrays["forces_cp2k"] = forces
                atoms_list.append(atoms)

            out_path = struct_dir / "extdata.xyz"
            write(str(out_path), atoms_list, format="extxyz")
            logger.info(f"[CP2K] Written {len(atoms_list)} frames → {out_path}")
            self.state[struct_key]["status"] = 1


# ---------------------------------------------------------------------------
# VASP stub
# ---------------------------------------------------------------------------

class VASPDFT(DFTBase):
    """Placeholder for a future VASP engine wrapper."""

    supports_reftraj = False

    def __init__(self, settings: dict, input_files: list[str], **kwargs) -> None:
        super().__init__("vasp", settings, input_files, **kwargs)

    def _prepare_structure(self, structures: list[str]) -> None:
        raise NotImplementedError("VASP support is not yet implemented.")

    def _run_calculation(self, structures: list[str]) -> None:
        raise NotImplementedError("VASP support is not yet implemented.")

    def _parse_outputs(self) -> None:
        raise NotImplementedError("VASP support is not yet implemented.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class DFTFactory:
    """
    Registry-based factory for DFT engine instances.

    Usage
    -----
        DFTFactory.register("myengine", MyEngineDFT)
        engine = DFTFactory.create("myengine", settings={...}, ...)
    """

    _registry: dict[str, type[DFTBase]] = {}

    @classmethod
    def register(cls, name: str, engine_cls: type[DFTBase]) -> None:
        cls._registry[name.lower()] = engine_cls

    @classmethod
    def create(cls, engine: str, **kwargs) -> DFTBase:
        """
        Instantiate a registered DFT engine.

        Raises
        ------
        ValueError
            If the engine name is unknown or not yet implemented.
        """
        key = engine.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown DFT engine '{engine}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key](**kwargs)


DFTFactory.register("cp2k", CP2KDFT)
DFTFactory.register("vasp", VASPDFT)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = DFTFactory.create(
        engine="cp2k",
        settings={"command": "mpirun -np 20 cp2k.popt"},
        input_files=["cp2k.inp", "force_eval.inp"],
        reftraj=True,
        run_mode="prepare",
        workdir="single_reftraj",
    )
    engine.run(["coord.xyz", "coord1.xyz"])
