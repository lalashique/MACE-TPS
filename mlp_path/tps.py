"""
tps.py — Transition Path Sampling (TPS) setup and sampling utilities.

Provides:
- ``ISetup``    — abstract base for OpenMM system construction.
- ``TPS_setup`` — concrete setup class (MACE potential via openmmml).
- ``ISampler``  — abstract base for path samplers.
- ``Sampler``   — TPS sampler using OpenPathSampling + MACE committee models.

Typical usage
-------------
    from tps import TPS_setup, Sampler

    setup   = TPS_setup("tps.json")
    sampler = Sampler(
        setup_class=setup,
        output_path="output",
        ops_storage_path="tps_storage.db",
        final_traj_storage_path="traj",
    )
    sampler.run()

CLI usage
---------
    python tps.py --start_type new        # fresh TPS run
    python tps.py --start_type from_db    # extract saved trajectories
"""

from __future__ import annotations

import argparse
import os
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Literal

import mdtraj
import numpy as np
import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
import torch
import yaml
from loguru import logger
from openpathsampling.experimental.storage import Storage, monkey_patch_all
from simtk import unit

import openmm as mm
from openmm import app
from openmm.app.element import Element
from openmm.app.topology import Topology
from openmmml import MLPotential
from openmmtools.integrators import VVVRIntegrator

from ase.io import read

from util.cv import CV
from util.parser import read_config, print_config, create_parser, global_arguments

# Apply OPS experimental-storage monkey patch globally
paths = monkey_patch_all(paths)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class ISetup(ABC):
    """
    Abstract base for OpenMM system construction.

    Reads the JSON config, initialises physical units, builds the OpenMM
    topology + system + MACE potential, and creates the OPS engine.

    Parameters
    ----------
    config_path : Path
        Path to the JSON config file.
    resource_directory : Path or str
        Directory containing auxiliary files (e.g. topology PDB).
    print_config_bool : bool
        Print the parsed config on init.
    """

    def __init__(
        self,
        config_path: Path,
        resource_directory: str | Path = "",
        print_config_bool: bool = True,
    ) -> None:
        self.resource_directory = str(resource_directory)
        self.config = read_config(config_path)
        if print_config_bool:
            print_config(self.config)

        # Physical units from config
        units_cfg        = self.config["UNITS"]
        self.size_unit   = getattr(unit, units_cfg["size_unit"])
        self.temp_unit   = getattr(unit, units_cfg["temp_unit"])
        self.time_unit   = getattr(unit, units_cfg["time_unit"])
        logger.debug(f"Units: size={self.size_unit}, temp={self.temp_unit}, time={self.time_unit}")

        # Integrator settings
        integ_cfg              = self.config["Integrator_settings"]
        self.temperature       = integ_cfg["temperature"] * self.temp_unit
        self.dt                = integ_cfg["dt"] * self.time_unit
        self.friction          = integ_cfg["friction"] / self.time_unit
        self.constant_variable = integ_cfg["constant_variable"]

        if self.constant_variable == "P":
            self.barostat_frequency = integ_cfg["barostat_frequency"]
            press_unit              = getattr(unit, units_cfg["press_unit"])
            self.pressure           = integ_cfg["pressure"] * press_unit
        else:
            if integ_cfg.get("barostat_frequency"):
                logger.warning(
                    "constant_variable is not 'P' — barostat settings will be ignored."
                )

        logger.debug(
            f"Integrator: T={self.temperature}, dt={self.dt}, friction={self.friction}"
        )

        # System identifiers
        self.system_name  = self.config["system_name"]
        self.platform_name = self.config["platform_name"]

        # MACE model path
        mace_path = self.config["mace_model_path"]
        self.mace_path = mace_path[0] if isinstance(mace_path, list) else mace_path

        # TPS sampling parameters
        tps_cfg               = self.config["TPS_settings"]
        self.n_frames_max     = tps_cfg["n_frames_max"]
        self.n_steps_per_frame = tps_cfg["n_steps_per_frame"]

        # Build the simulation system
        self.integrator_temp          = self._setup_integrator()
        self._topology, self.system, self.positions = self._setup_system()
        self.template                 = self._create_template()
        self.topology                 = self.template.topology
        self.integrator               = self._setup_integrator()
        self.md_engine                = self._setup_engine(self.integrator)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_integrator(self) -> VVVRIntegrator:
        logger.debug("Initialising VVVR integrator.")
        integrator = VVVRIntegrator(self.temperature, self.friction, self.dt)
        integrator.setConstraintTolerance(1e-5)
        return integrator

    def _setup_engine(self, integrator) -> ops_openmm.Engine:
        logger.debug(f"Initialising OPS engine on {self.platform_name}.")
        platform = mm.Platform.getPlatformByName(self.platform_name)
        engine = ops_openmm.Engine(
            topology=self.topology,
            system=self.system,
            integrator=integrator,
            openmm_properties=self._platform_properties(),
            options={
                "n_frames_max":     self.n_frames_max,
                "n_steps_per_frame": self.n_steps_per_frame,
            },
        )
        engine.name = "default"
        engine.initialize(platform)
        return engine

    def _setup_system(self) -> tuple:
        """Build OpenMM topology, MACE system, and initial positions."""
        logger.debug("Building OpenMM topology and MACE system.")
        top_file = self.config.get("top_file", "top.pdb")
        atoms    = read(top_file)
        pos      = atoms.get_positions() / 10   # Å → nm
        box      = atoms.get_cell() / 10         # Å → nm
        elements = atoms.get_chemical_symbols()

        topology = Topology()
        for element, position in zip(elements, pos):
            e     = Element.getBySymbol(element)
            chain = topology.addChain(element)
            res   = topology.addResidue(element, chain)
            topology.addAtom(element, e, res, id=element)

        if max(atoms.get_cell().cellpar()[:3]) > 0:
            topology.setPeriodicBoxVectors(vectors=box)

        potential = MLPotential("mace", modelPath=self.mace_path, device="cuda")
        system    = potential.createSystem(topology, nl="torch", max_n_pairs=-1)

        if self.constant_variable == "P":
            system.addForce(
                mm.MonteCarloBarostat(self.pressure, self.temperature, self.barostat_frequency)
            )

        sim = app.Simulation(topology, system, self.integrator_temp)
        sim.context.setPositions(pos)
        new_box_vectors = sim.context.getState(getParameters=True).getPeriodicBoxVectors()
        sim.topology.setPeriodicBoxVectors(new_box_vectors)
        positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)

        return topology, system, positions

    def _platform_properties(self) -> dict:
        return {
            "OpenCL": {"OpenCLPrecision": "mixed", "OpenCLDeviceIndex": "2"},
            "CUDA":   {"CudaPrecision": "mixed"},
            "CPU":    {},
        }.get(self.platform_name, {})

    def _create_template(self):
        logger.debug("Creating OPS snapshot template.")
        LocalSystem = namedtuple("LocalSystem", ["name", "system", "topology", "positions"])
        test_sys    = LocalSystem(
            name="system", system=self.system,
            topology=self._topology, positions=self.positions,
        )
        return ops_openmm.snapshot_from_testsystem(test_sys)


class ISampler(ABC):
    """Abstract base for path samplers."""

    @abstractmethod
    def run(self) -> None: ...


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class Sampler(ISampler):
    """
    TPS sampler built on OpenPathSampling.

    Loads an initial trajectory, constructs a TPS network from the CVs defined
    in the config, runs the Monte Carlo sampling loop, and optionally saves
    accepted (and rejected) trajectories to disk at regular intervals.

    Parameters
    ----------
    setup_class : ISetup
        An already-initialised system setup object.
    directory : str or Path
        Working directory (used for config resolution).
    output_path : str or Path, optional
        Directory for OPS storage and trajectory files.
    ops_storage_path : str
        Filename for the main OPS ``.db`` storage.
    save_final_traj : bool
        Whether to save the last accepted trajectory as a separate storage.
    final_traj_storage_path : str or Path, optional
        Directory to write per-step XYZ trajectory files.
    system_resource_directory : str or Path
        Location of auxiliary system files.
    """

    def __init__(
        self,
        setup_class: ISetup,
        directory: str | Path = ".",
        output_path: str | Path | None = None,
        ops_storage_path: str = "tps_storage.db",
        save_final_traj: bool = False,
        final_traj_storage_path: str | Path | None = None,
        system_resource_directory: str | Path = "",
        # Legacy parameters kept for backwards compatibility
        prepare_only: bool = False,
        absolute_paths: bool = False,
        pickle_path: str | Path | None = None,
    ) -> None:
        self.setup_class               = setup_class
        self.directory                 = Path(directory)
        self.config_file               = setup_class.config
        self.n_steps                   = self.config_file.get("tot_mc_steps", 100)
        self.step                      = self.config_file.get("save_every", None)
        self.traj_path                 = Path(self.config_file.get("init_path", "tps.xyz"))
        self.output_path               = Path(output_path) if output_path else self.directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.ops_storage_path          = ops_storage_path
        self.save_final_traj           = save_final_traj
        self.final_traj_storage_path   = Path(final_traj_storage_path) if final_traj_storage_path else None
        self.system_resource_directory = str(system_resource_directory)
        # Legacy — not used in the current implementation but kept for API compatibility
        self.prepare_only  = prepare_only
        self.absolute_paths = absolute_paths
        self.pickle_path   = pickle_path

        if self.step is None:
            logger.warning(
                f"'save_every' not set — trajectories will only be saved "
                f"at the end of the full {self.n_steps}-step run."
            )

    # ------------------------------------------------------------------
    # Pickle support (engine objects cannot be pickled)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in ("setup_class", "engine", "template"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the TPS sampling loop."""
        setup = self.setup_class
        paths.PathMover.engine = setup.md_engine

        # Load initial trajectory
        init_traj = self._load_trajectory(self.traj_path, setup)

        # Build the OPS network and move scheme
        ops_storage_path = self.output_path / (
            self.ops_storage_path or f"d_tps_storage_{setup.system_name}.db"
        )
        logger.debug(f"OPS storage path: {ops_storage_path}")
        logger.debug(f"MD engine timestep: {setup.md_engine.simulation.integrator.getStepSize()}")

        cv        = CV(self.config_file, self.traj_path)
        network   = paths.TPSNetwork.from_states_all_to_all(cv.states)
        selector  = paths.shooting.UniformSelector()
        beta      = 1.0 / (setup.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
        modifier  = paths.RandomVelocities(beta=beta, engine=setup.md_engine)

        tw_strategy = paths.strategies.TwoWayShootingStrategy(
            modifier=modifier, selector=selector, engine=setup.md_engine, group="TwoWayShooting"
        )
        move_scheme = paths.MoveScheme(network=network)
        move_scheme.append(tw_strategy)
        move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
        move_scheme.build_move_decision_tree()

        initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
        initial_conditions.sanity_check()

        storage = Storage(ops_storage_path, "w")
        sampler = paths.PathSampling(storage, move_scheme, initial_conditions).named("sampler")

        save_rejected = self.config_file.get("save_rejected", False)

        if self.step is not None:
            self._run_chunked(sampler, storage, save_rejected)
        else:
            logger.debug(f"Running {self.n_steps} TPS steps in one block.")
            sampler.run(self.n_steps)

        # Save initial accepted path
        self._try_save_initial_path(storage)

        # Optionally save final trajectory
        if self.save_final_traj and self.final_traj_storage_path:
            final_db = self.output_path / f"tps_final_{setup.system_name}.db"
            final_storage = Storage(final_db, "w")
            final_storage.save(storage.steps[-1].active[0].trajectory)
            final_storage.close()
            logger.info(f"Final trajectory saved → {final_db}")

        storage.close()

    def _run_chunked(self, sampler, storage, save_rejected: bool) -> None:
        """Run TPS in chunks of ``self.step`` steps, saving trajectories periodically."""
        traj_dir = self.final_traj_storage_path
        if traj_dir:
            traj_dir.mkdir(parents=True, exist_ok=True)

        for chunk_start in range(0, self.n_steps + 1, self.step):
            logger.debug(f"Running steps {chunk_start} → {chunk_start + self.step}")
            sampler.run(self.step)

            if traj_dir is None:
                continue

            for j, step in enumerate(storage.steps):
                if j <= chunk_start:
                    continue
                traj = storage.trajectories[j]
                if step.change.accepted:
                    traj.to_mdtraj().save(str(traj_dir / f"accepted_{j}.xyz"))
                    logger.debug(f"Saved accepted trajectory for step {j}.")
                elif save_rejected:
                    traj.to_mdtraj().save(str(traj_dir / f"rejected_{j}.xyz"))

    def _try_save_initial_path(self, storage) -> None:
        if self.final_traj_storage_path:
            try:
                self.final_traj_storage_path.mkdir(parents=True, exist_ok=True)
                storage.steps[0].active[0].trajectory.to_mdtraj().save(
                    str(self.final_traj_storage_path / "accepted_0.xyz")
                )
            except Exception as exc:
                logger.error(f"Failed to save initial path: {exc}")

    # ------------------------------------------------------------------
    # Save from existing storage (post-hoc extraction)
    # ------------------------------------------------------------------

    def save_from_storage(self, storage_path: str | Path, out_folder: str | Path) -> None:
        """
        Extract accepted and rejected trajectories from an existing OPS storage.

        This is a convenience wrapper around ``util.storage_helper.save_from_storage``.

        Parameters
        ----------
        storage_path : str or Path
            OPS ``.db`` storage file.
        out_folder : str or Path
            Directory to write XYZ trajectory files.
        """
        from util.storage_helper import save_from_storage as _save
        _save(storage_path, self.config_file, out_folder)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load_trajectory(traj_path: Path, setup: ISetup):
        """Load an initial trajectory from .nc, .db, or .xyz file."""
        top_file = setup.config.get("top_file", "top.pdb")

        if traj_path.suffix == ".nc":
            old   = paths.Storage(traj_path)
            traj  = old.tag["trajectory"]
            mdt   = traj.to_mdtraj()
            old.close()
            return ops_openmm.trajectory_from_mdtraj(mdt)

        if traj_path.suffix == ".db":
            old  = Storage(traj_path)
            traj = old.trajectories[-1]
            mdt  = traj.to_mdtraj()
            old.close()
            return ops_openmm.trajectory_from_mdtraj(mdt)

        if traj_path.suffix == ".xyz":
            top  = mdtraj.load(top_file)
            traj = mdtraj.load(str(traj_path), top=top)
            traj.unitcell_vectors = np.array(list(top.unitcell_vectors) * len(traj.xyz))
            return ops_openmm.trajectory_from_mdtraj(
                traj, velocities=[np.zeros_like(f) for f in traj.xyz]
            )

        raise ValueError(
            f"Unsupported trajectory format: '{traj_path.suffix}'. "
            "Use .nc, .db, or .xyz."
        )

    @staticmethod
    def path_extension(
        traj,
        n_steps: int,
        engine,
        direction: list[str] | None = None,
    ):
        """
        Extend a trajectory by ``n_steps`` forward and/or backward.

        Parameters
        ----------
        traj : OPS trajectory
        n_steps : int
        engine : OPS engine
        direction : list of {"forward", "backward"}, optional
        """
        last_frame = traj.frames[-1]
        engine.current_snapshot = last_frame
        extended = engine.generate_trajectory(n_steps=n_steps)
        return traj + extended


# ---------------------------------------------------------------------------
# Concrete TPS setup
# ---------------------------------------------------------------------------

class TPS_setup(ISetup):
    """
    TPS system setup using a MACE potential via openmmml.

    Inherits all setup logic from ``ISetup``.  The JSON config must contain
    at minimum: ``UNITS``, ``Integrator_settings``, ``TPS_settings``,
    ``mace_model_path``, ``platform_name``, ``system_name``, and ``top_file``.
    """

    def __init__(
        self,
        config_path: Path | str,
        resource_directory: Path | str = ".",
        print_config_bool: bool = True,
    ) -> None:
        super().__init__(Path(config_path), resource_directory, print_config_bool)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TPS sampling.")
    parser.add_argument(
        "--start_type",
        choices=["new", "from_db", "continue"],
        default="new",
        help="'new': fresh run. 'from_db': extract trajectories from existing storage.",
    )
    args, _ = parser.parse_known_args()

    tps   = TPS_setup("tps.json")
    sampler = Sampler(
        setup_class=tps,
        directory=".",
        output_path="output",
        ops_storage_path="tps_storage.db",
        final_traj_storage_path=Path("traj"),
        system_resource_directory=".",
    )

    if args.start_type == "from_db":
        logger.info("Extracting trajectories from existing OPS storage.")
        sampler.save_from_storage(
            sampler.output_path / sampler.ops_storage_path,
            sampler.final_traj_storage_path,
        )
    elif args.start_type == "new":
        logger.info("Starting new TPS run.")
        sampler.run()
    elif args.start_type == "continue":
        logger.info("'continue' mode not yet implemented.")
    else:
        raise ValueError(f"Unknown start_type: {args.start_type}")
