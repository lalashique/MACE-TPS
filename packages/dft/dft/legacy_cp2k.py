"""Ported almost verbatim from the original mlp_path/dft.py.

This is NOT a rewrite - it's the existing CP2K/reftraj logic, moved here so
it lives behind the new `DFTEngine` interface (see cp2k_backend.py) instead
of being imported ad hoc by workflow.py. The DFT owner should keep iterating
on this file; known rough edges carried over from the original:

  - `_parse_outputs` originally imported `cpoker.iparser`/`cpoker.genfun`,
    an unpublished private dependency - fixed. The cell-lookup part
    (`iparser`, which resolves CP2K's `@SET`/`@include` directives) is now
    vendored in cpoker_iparser.py; the energy/force parsing was rewritten
    from scratch without `genfun`. See `_parse_outputs`'s own docstring for
    the full story.
  - `_run_calculation` doesn't actually invoke CP2K; execution is still a
    manual step (the printed `command`) - see NeedsExternalJob usage in
    cp2k_backend.py, which is the new, explicit way of expressing that.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Literal
import os
from loguru import logger


class DFTBase(ABC):
    """Abstract base class for all DFT engines."""

    supports_reftraj: bool = False

    def __init__(
        self,
        engine: str,
        settings: dict,
        input_files: list,
        reftraj: bool = False,
        run_mode: Literal["prepare", "run"] = "prepare",
        workdir: str = "output",
    ):
        self.engine = engine
        self.settings = settings
        self.reftraj = reftraj
        self.run_mode = run_mode
        self.workdir = Path(workdir)
        self.input_files = [Path(".") / f for f in input_files]
        self.completed = False
        self.results = None

        self.state = {"engine": self.engine, "run_mode": self.run_mode}
        self._validate_capabilities()

    def _validate_capabilities(self):
        if self.reftraj and not self.supports_reftraj:
            raise ValueError(
                f"[{self.engine.upper()}] reftraj mode is not supported. "
                f"Set reftraj=False or choose CP2K."
            )

    def run(self, structures):
        logger.debug(f"[BASE] Current state: {self.state}")
        self._prepare_structure(structures)
        if self.run_mode == "prepare":
            logger.debug(f"[BASE] {self.engine}: inputs prepared (mode={self._mode_name()}).")
            self._save_state()
            return
        self._run_calculation(structures)
        self._parse_outputs()
        self.completed = True
        self._save_state()

    @abstractmethod
    def _prepare_structure(self, structures): ...
    @abstractmethod
    def _run_calculation(self, structures): ...
    @abstractmethod
    def _parse_outputs(self): ...

    def _mode_name(self):
        return "reftraj" if self.reftraj else "single"

    def _save_state(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        with open(self.workdir / "state.json", "w") as f:
            json.dump(self.state, f, indent=2)

    def _load_state(self):
        state_file = self.workdir / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")
        with open(state_file, "r") as f:
            return json.load(f)


class CP2KDFT(DFTBase):
    supports_reftraj = True

    def __init__(self, settings, input_files, reftraj=False, run_mode="run", workdir="dft_cp2k"):
        super().__init__("cp2k", settings, input_files, reftraj, run_mode, workdir)

    def _prepare_structure(self, structures):
        input_dir = self.workdir
        input_dir.mkdir(parents=True, exist_ok=True)
        if self.reftraj:
            self._write_reftraj_input(input_dir, structures)
        else:
            raise NotImplementedError("Non-reftraj CP2K mode was never implemented in the original code.")

    def _write_reftraj_input(self, input_dir, structures):
        for i, struct in enumerate(structures):
            struct = Path(struct).resolve()
            input_dir_i = input_dir / f"structure_{i}"
            input_dir_i.mkdir(parents=True, exist_ok=True)
            for file in self.input_files:
                dest_file = input_dir_i / file.name
                dest_file.write_text(Path(file).read_text())
            coord_file = input_dir_i / "coord.xyz"
            coord_file.write_bytes(struct.read_bytes())

            pause_file = input_dir_i / "PAUSE"
            pause_file.write_text("")

            logger.debug(f"[CP2K] Preparing structure: {struct}")
            # Store the full resolved path, not just the basename
            # ("structure_0") - _parse_outputs does Path(struct_key).resolve(),
            # which against a bare basename resolves relative to whatever the
            # CURRENT process cwd happens to be at parse time, not relative to
            # self.workdir. That mismatch is what made _parse_outputs look for
            # cp2k-pos-1.xyz in the wrong directory entirely.
            self.state[str(input_dir_i.resolve())] = {"status": 0, "file": str(struct)}
            logger.info(
                f"Run using command: \n cd {input_dir_i} && "
                f"{self.settings['command']} -i {self.input_files[0]} -o cp2k.out"
            )

    def _run_calculation(self, structures):
        logger.debug(f"[CP2K] Running DFT calculation in {self.workdir}...")
        # Execution is external - see cp2k_backend.CP2KDFTEngine.label, which
        # raises NeedsExternalJob with the exact command to run instead of
        # silently doing nothing here.

    def _parse_outputs(self):
        """Parse cp2k-pos-1.xyz/cp2k-frc-1.xyz directly (ase/numpy + a
        vendored CP2K-input parser for the cell).

        The original implementation imported `cpoker.iparser`/`cpoker.genfun`
        - not a published dependency, so this failed until it was replaced.
        Once the user made the real `cpoker` library available (private,
        at ~/Library/CloudStorage/OneDrive-UvA/scriptarium/pylib/pylib/cpoker/),
        its `iparser` class - the piece that actually mattered here, since it
        resolves `@SET`/`@include` in a CP2K deck - was vendored (trimmed,
        no `genfun` dependency) into cpoker_iparser.py and is used below for
        the cell. `genfun.py` was NOT vendored: its only use in the original
        was `unconv()`'s grep/awk convergence check, which this rewrite
        already replaced with the energies parsed straight from
        cp2k-pos-1.xyz's comment lines; pulling in `genfun` (and its
        `numba`/`uncertainties` dependencies) just for that would add
        dependencies for no benefit. Verified against a real CP2K run (tiny
        water reftraj, see real_test/dft_real) - the Hartree->eV and
        Hartree/Bohr->eV/Angstrom conversion factors are unchanged from the
        original.

        Simplification relative to the original: no convergence-flag
        filtering (grep "converge" from cp2k.out) - every reftraj frame is
        kept as-is. Re-add that filtering here if silently including
        unconverged SCF steps in training data becomes a real concern.
        """
        import re
        import numpy as np
        from ase import Atoms
        from ase.io import write

        from dft.cpoker_iparser import read_cell_abc

        HARTREE_TO_EV = 27.211386245988
        HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.42206747632590

        logger.debug(f"[CP2K] Parsing outputs in {self.workdir}...")
        for struct_key in self.state:
            if "structure_" not in struct_key:
                continue
            folder = Path(struct_key).resolve()

            cell = read_cell_abc(folder)

            pos_lines = (folder / "cp2k-pos-1.xyz").read_text().splitlines()
            frc_lines = (folder / "cp2k-frc-1.xyz").read_text().splitlines()

            converge_flags = None
            if self.settings.get("filter_unconverged", False):
                # Legacy behaviour: one "converge" line per reftraj step in cp2k.out;
                # steps whose line contains "NOT" are dropped.
                out_lines = [l for l in (folder / "cp2k.out").read_text().splitlines() if "converge" in l]
                converge_flags = ["NOT" not in l for l in out_lines]
                logger.info(f"[CP2K] converged steps: {sum(converge_flags)}/{len(converge_flags)} in {folder}")

            atoms_list = []
            frame_idx = 0
            i = 0
            while i < len(pos_lines):
                if not pos_lines[i].strip():
                    i += 1
                    continue
                n = int(pos_lines[i].strip())
                comment = pos_lines[i + 1]
                e_match = re.search(r"E\s*=\s*([-\d.eE]+)", comment)
                if e_match is None:
                    raise ValueError(f"No energy found in comment line: {comment!r}")
                energy = float(e_match.group(1)) * HARTREE_TO_EV

                symbols, positions, forces = [], [], []
                for j in range(n):
                    parts = pos_lines[i + 2 + j].split()
                    symbols.append(parts[0])
                    positions.append([float(x) for x in parts[1:4]])
                for j in range(n):
                    parts = frc_lines[i + 2 + j].split()
                    forces.append([float(x) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM for x in parts[1:4]])

                atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=False)
                atoms.info["energy_cp2k"] = energy
                atoms.arrays["forces_cp2k"] = np.array(forces)
                if converge_flags is None or (frame_idx < len(converge_flags) and converge_flags[frame_idx]):
                    atoms_list.append(atoms)
                frame_idx += 1
                i += 2 + n

            if converge_flags is not None and len(converge_flags) != frame_idx:
                raise ValueError(
                    f"{folder}: {len(converge_flags)} 'converge' lines in cp2k.out but {frame_idx} frames in cp2k-pos-1.xyz"
                )
            write(str(folder / "extdata.xyz"), atoms_list, format="extxyz")


class DFTFactory:
    registry = {}

    @classmethod
    def register(cls, engine_name, cls_ref):
        cls.registry[engine_name.lower()] = cls_ref

    @classmethod
    def create(cls, engine, **kwargs):
        engine = engine.lower()
        if engine not in cls.registry:
            raise ValueError(f"Unknown DFT engine '{engine}'. Available: {list(cls.registry.keys())}")
        return cls.registry[engine](**kwargs)


DFTFactory.register("cp2k", CP2KDFT)
