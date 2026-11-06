"""Ported from the original mlp_path/tps.py.

`ISetup`, `ISampler`, `Sampler`, and `TPS_setup` are moved here unchanged in
spirit (OpenPathSampling/OpenMM setup + two-way shooting run loop).

`extend_and_select_frames()` is new: the original file only had this logic
inline in an `if __name__ == "__main__":` block, which meant it could only
be run as a script and could never be called from workflow.py. It's the
same logic (evaluate committee variance along each accepted trajectory,
optionally extend both ends with free MD, merge, threshold-filter, and
weighted-sample the highest-variance frames), just turned into a function
so `ops_backend.py` can call it directly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import Storage, monkey_patch_all

paths = monkey_patch_all(paths)

from openmm import app
import openmm as mm
from openmmtools.integrators import VVVRIntegrator
from simtk import unit
import openpathsampling.engines.openmm as eng
from collections import namedtuple

from openmmml import MLPotential
from ase.io import read, write
from openmm.app.topology import Topology
from openmm.app.element import Element

import mdtraj

from tps.util.parser import read_config, print_config
from tps.util import CV

from abc import ABC, abstractmethod
from loguru import logger


class ISetup(ABC):
    def __init__(self, config_path, resource_directory="", print_config_bool=True):
        self.resource_directory = resource_directory
        self.config = read_config(config_path)
        if print_config_bool:
            print_config(self.config)

        self.size_unit = getattr(unit, self.config["UNITS"]["size_unit"])
        self.temp_unit = getattr(unit, self.config["UNITS"]["temp_unit"])
        self.time_unit = getattr(unit, self.config["UNITS"]["time_unit"])

        integrator_config = self.config.get("Integrator_settings")
        self.temperature = integrator_config["temperature"] * self.temp_unit
        self.dt = integrator_config["dt"] * self.time_unit
        self.friction = integrator_config["friction"] / self.time_unit
        self.constant_variable = integrator_config["constant_variable"]
        if self.constant_variable == "P":
            self.barostat_frequency = integrator_config["barostat_frequency"]
            self.pressure = integrator_config["pressure"] * self.press_unit

        self.system_name = self.config["system_name"]
        self.platform_name = self.config["platform_name"]

        self.mace_path = self.config["mace_model_path"]
        if isinstance(self.mace_path, list):
            self.mace_path = self.mace_path[0]

        self.TPS_settings = self.config.get("TPS_settings")
        self.n_frames_max = self.TPS_settings["n_frames_max"]
        self.n_steps_per_frame = self.TPS_settings["n_steps_per_frame"]

        self.integrator_temp = self._setup_integrator()
        self._topology, self.system, self.positions = self._setup_system()
        self.template = self._create_template()
        self.topology = self.template.topology
        self.integrator = self._setup_integrator()
        self.md_engine = self._setup_engine(self.integrator)

    def _setup_integrator(self):
        integrator = VVVRIntegrator(self.temperature, self.friction, self.dt)
        integrator.setConstraintTolerance(0.00001)
        return integrator

    def _setup_engine(self, integrator):
        platform = mm.Platform.getPlatformByName(self.platform_name)
        engine_options = {
            "n_frames_max": self.n_frames_max,
            "n_steps_per_frame": self.n_steps_per_frame,
        }
        engine = eng.Engine(
            topology=self.topology,
            system=self.system,
            integrator=integrator,
            openmm_properties=self._get_platform_properties(),
            options=engine_options,
        )
        engine.name = "default"
        engine.initialize(platform)
        return engine

    def _setup_system(self):
        atoms = read(self.config.get("top_file", "top.pdb"))
        pos = atoms.get_positions() / 10
        box_vectors = atoms.get_cell() / 10
        elements = atoms.get_chemical_symbols()

        topology = Topology()
        for element, position in zip(elements, pos):
            e = Element.getBySymbol(element)
            chain = topology.addChain(element)
            res = topology.addResidue(element, chain)
            topology.addAtom(element, e, res, id=element)
        if max(atoms.get_cell().cellpar()[:3]) > 0:
            topology.setPeriodicBoxVectors(vectors=box_vectors)

        # Original hardcoded device="cuda" here, which made this unrunnable
        # on any machine without an NVIDIA GPU (confirmed while testing this
        # workbench on Apple Silicon - no CUDA). torch device names don't
        # match OpenMM platform names 1:1 (e.g. "OpenCL" isn't a torch
        # device), so map the ones that do have a sane torch equivalent and
        # otherwise fall back to cpu.
        torch_device = {"CUDA": "cuda", "CPU": "cpu"}.get(self.platform_name, "cpu")
        prmtop = MLPotential("mace", modelPath=self.mace_path, device=torch_device)
        system = prmtop.createSystem(topology, nl="torch", max_n_pairs=-1)

        if self.constant_variable == "P":
            system.addForce(mm.MonteCarloBarostat(self.pressure, self.temperature, self.barostat_frequency))

        init_simulation = app.Simulation(topology, system, self.integrator_temp)
        init_simulation.context.setPositions(pos)
        new_box_vectors = init_simulation.context.getState(getParameters=True).getPeriodicBoxVectors()
        init_simulation.topology.setPeriodicBoxVectors(new_box_vectors)
        positions = init_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        return topology, system, positions

    def _get_platform_properties(self):
        if self.platform_name == "OpenCL":
            return {"OpenCLPrecision": "mixed", "OpenCLDeviceIndex": "2"}
        if self.platform_name == "CUDA":
            return {"CudaPrecision": "mixed"}
        return {}

    def _create_template(self):
        testsystem = namedtuple("LocalTestSystem", ["name", "system", "topology", "positions"])
        test_system = testsystem(name="CB7:B2", system=self.system, topology=self._topology, positions=self.positions)
        return eng.snapshot_from_testsystem(test_system)


class ISampler(ABC):
    @abstractmethod
    def run(self): ...


class Sampler(ISampler):
    def __init__(
        self,
        setup_class: ISetup,
        directory=".",
        output_path=None,
        ops_storage_path="tps_storage.db",
        save_final_traj=False,
        final_traj_storage_path=None,
        system_resource_directory="",
    ):
        self.setup_class = setup_class
        self.directory = Path(directory)
        self.config_file = setup_class.config
        self.n_steps = self.config_file.get("tot_mc_steps", 100)
        self.step = self.config_file.get("save_every", None)
        self.traj_path = Path(self.config_file.get("init_path", "tps.xyz"))
        self.output_path = Path(output_path) if output_path else self.directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.ops_storage_path = ops_storage_path
        self.save_final_traj = save_final_traj
        self.final_traj_storage_path = final_traj_storage_path
        self.system_resource_directory = system_resource_directory

    def run(self):
        setup = self.setup_class
        paths.PathMover.engine = setup.md_engine

        traj_path = self.traj_path
        if traj_path.suffix == ".xyz":
            mdt = mdtraj.load(self.config_file.get("top_file", "top.pdb"))
            traj = mdtraj.load(traj_path, top=mdt)
            traj.unitcell_vectors = np.array(list(mdt.unitcell_vectors) * len(traj.xyz))
            init_traj = ops_openmm.trajectory_from_mdtraj(
                traj, velocities=[np.zeros_like(frame) for frame in traj.xyz]
            )
        else:
            raise ValueError("Unsupported trajectory format (only .xyz is wired up in this port)")

        ops_storage_path = self.output_path / (self.ops_storage_path or f"d_tps_storage_{setup.system_name}.db")

        selector = paths.shooting.UniformSelector()
        beta = 1.0 / (setup.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
        modifier = paths.RandomVelocities(beta=beta, engine=setup.md_engine)
        tw_strategy = paths.strategies.TwoWayShootingStrategy(
            modifier=modifier, selector=selector, engine=setup.md_engine, group="TwoWayShooting"
        )

        cv = CV(self.config_file, self.traj_path)
        network = paths.TPSNetwork.from_states_all_to_all(cv.states)

        move_scheme = paths.MoveScheme(network=network)
        move_scheme.append(tw_strategy)
        move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
        move_scheme.build_move_decision_tree()

        initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
        initial_conditions.sanity_check()

        storage = Storage(ops_storage_path, "w")
        sampler = paths.PathSampling(storage, move_scheme, initial_conditions).named("sampler")

        save_rejected = self.config_file.get("save_rejected", False)
        os.makedirs(self.final_traj_storage_path, exist_ok=True)

        if self.step is not None:
            for i in range(0, self.n_steps + 1, self.step):
                sampler.run(self.step)
                for (j, s), _ in zip(enumerate(storage.steps), storage.trajectories):
                    if j <= i:
                        continue
                    if s.change.accepted:
                        storage.trajectories[j].to_mdtraj().save(f"{self.final_traj_storage_path}/accepted_{j}.xyz")
                    elif save_rejected:
                        storage.trajectories[j].to_mdtraj().save(f"{self.final_traj_storage_path}/rejected_{j}.xyz")
        else:
            sampler.run(self.n_steps)

        try:
            storage.steps[0].active[0].trajectory.to_mdtraj().save(f"{self.final_traj_storage_path}/accepted_0.xyz")
        except Exception as e:
            logger.error(f"Failed to save initial path: {e}")

        if self.save_final_traj:
            final_path = self.output_path / (self.final_traj_storage_path or f"tps_final_{setup.system_name}.db")
            final_storage = Storage(final_path, "w")
            final_storage.save(storage.steps[-1].active[0].trajectory)
            final_storage.close()
        storage.close()


class TPS_setup(ISetup):
    def __init__(self, config_path: Path, resource_directory: Path = Path("."), print_config_bool=True):
        super().__init__(config_path, resource_directory, print_config_bool)


def extend_and_select_frames(sampler: Sampler, mace_model_paths: list, workdir: Path) -> Path:
    """Evaluate committee uncertainty along accepted TPS trajectories,
    optionally extend both ends with free MD, merge, threshold-filter, and
    weighted-sample the highest-uncertainty frames.

    Returns the path to the final selected-frames extxyz file, ready to be
    sent to the DFT stage of the next generation.
    """
    import ase.units as units
    from mace.calculators import MACECalculator
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.langevin import Langevin
    from tps.util.mlp_helper import paper_selection_weights, process_frame
    from mlp.legacy_mace import MACE_MLP

    workdir = Path(workdir)
    nsteps = sampler.config_file.get("extend_steps", None)
    timestep = sampler.config_file.get("Integrator_settings")["dt"] * 1000
    temp = sampler.config_file.get("Integrator_settings")["temperature"]
    friction = sampler.config_file.get("Integrator_settings")["friction"] / 1000

    if not isinstance(mace_model_paths, list):
        raise ValueError("mace_model_paths should be a list of model paths for committee evaluation.")
    # Same platform_name -> torch device mapping as ISetup._setup_system -
    # blindly lowercasing platform_name (the original code) breaks for
    # "OpenCL" (not a torch device) and defaults to "cuda" when unset, which
    # fails outright on non-NVIDIA machines.
    torch_device = {"CUDA": "cuda", "CPU": "cpu"}.get(sampler.config_file.get("platform_name"), "cpu")
    committee_calcs = [MACECalculator(model_paths=[m], device=torch_device) for m in mace_model_paths]
    evaluate_committee = MACE_MLP.evaluate_committee

    folder = Path(sampler.final_traj_storage_path).resolve()
    traj_list = sorted(
        (folder / f for f in os.listdir(folder) if f.endswith(".xyz") and not f.startswith("._")),
        key=lambda x: int(x.stem.split("_")[-1].split(".")[0]),
    )

    cell = read(Path(sampler.config_file.get("top_file", "top.pdb")).resolve()).get_cell()
    extend_folder = folder / "extend"
    extend_folder.mkdir(parents=True, exist_ok=True)

    for traj_file in traj_list:
        base_name = traj_file.name.replace(".xyz", "")
        if (extend_folder / f"{base_name}_extended.xyz").exists():
            continue

        orig_traj = read(f"{traj_file}", index=":")
        atoms_first, atoms_last = orig_traj[0].copy(), orig_traj[-1].copy()

        frames = []
        for atoms in orig_traj:
            atoms = atoms.copy()
            atoms.cell = cell
            atoms.set_pbc((True, True, True))
            atoms.calc = committee_calcs[0]
            evaluate_committee(atoms, committee_calcs=committee_calcs)
            frames.append(atoms.copy())
        write(extend_folder / f"{base_name}.xyz", frames)
        frames.clear()

        if "accepted" not in base_name:
            continue

        if not nsteps:
            write(extend_folder / f"{base_name}_extended.xyz", read(extend_folder / f"{base_name}.xyz", index=":"))
            continue

        for direction, atoms in zip(["backward", "forward"], [atoms_first, atoms_last]):
            atoms.cell = cell
            atoms.set_pbc((True, True, True))
            atoms.calc = committee_calcs[0]
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
            if direction == "backward":
                atoms.set_velocities(-atoms.get_velocities())

            out_file = extend_folder / f"{base_name}_{direction}.xyz"
            if out_file.exists():
                out_file.unlink()

            frames = []
            dyn = Langevin(atoms, timestep * units.fs, temperature_K=temp, friction=friction)

            def buffer_frame():
                evaluate_committee(atoms, committee_calcs=committee_calcs)
                frames.append(atoms.copy())

            dyn.attach(buffer_frame, interval=1)
            dyn.run(nsteps)
            for a in frames:
                a.arrays.pop("momenta", None)
            write(out_file, frames)
            frames.clear()

        backward = read(extend_folder / f"{base_name}_backward.xyz", index=":")[::-1]
        forward = read(extend_folder / f"{base_name}_forward.xyz", index=":")
        orig = read(extend_folder / f"{base_name}.xyz", index=":")
        write(extend_folder / f"{base_name}_extended.xyz", backward + orig + forward)

    combined = []
    extend_traj_list = sorted(
        (
            extend_folder / f
            for f in os.listdir(extend_folder)
            if (f.endswith("_extended.xyz") or f.startswith("rejected")) and not f.startswith("._")
        ),
        key=lambda x: int(x.stem.strip("_extended.xyz").split("_")[-1]),
    )
    for traj_file in extend_traj_list:
        combined.extend(read(traj_file, index=":"))
    combined_out = workdir / "tps_all_extended.xyz"
    write(combined_out, combined)

    cutoff = sampler.config_file.get("local_cutoff", 5.0)
    carbon_indices = sampler.config_file.get("selection_indices")
    num_select = sampler.config_file.get("num_select")
    e_thresh = sampler.config_file.get("e_var_max", 1000)
    f_thresh = sampler.config_file.get("f_var_max", 100)
    alpha = sampler.config_file.get("alpha", 0)

    atoms_list = read(combined_out, index=":")
    metrics = np.array([process_frame(at, carbon_indices, cutoff) for at in atoms_list])

    mask = (metrics[:, 0] <= e_thresh) & (metrics[:, 1] <= f_thresh)
    metrics_filt, atoms_filt = metrics[mask], [a for a, keep in zip(atoms_list, mask) if keep]
    if len(metrics_filt) == 0:
        metrics_filt, atoms_filt = metrics, atoms_list

    scheme = sampler.config_file.get("selection_scheme", "legacy")
    if scheme == "paper":
        # Lal et al. (arXiv:2605.03737) Eq. 2-4: per-observable variance normalised over the dataset
        # (energy, force, CO2-local force), summed into S_i; importance weights w_i ~ lam*S_i + (1-lam).
        lam = float(sampler.config_file.get("lambda", 0.7))
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {lam}")
        w = paper_selection_weights(metrics_filt, lam)
        n_pick = min(num_select, len(w))
        pick = np.random.choice(len(w), size=n_pick, replace=False, p=w / w.sum())
        selected = [atoms_filt[i] for i in np.sort(pick)]
        output_path = workdir / "tps_selected_frames.xyz"
        write(output_path, selected)
        return output_path
    if scheme != "legacy":
        raise ValueError(f"Unknown selection_scheme: {scheme!r} (use 'legacy' or 'paper')")

    worst_idx = []
    for col in range(metrics_filt.shape[1]):
        vals = metrics_filt[:, col]
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        weights = alpha * vals_norm + (1 - alpha)
        probs = weights / weights.sum()
        worst_idx.append(np.random.choice(len(vals), size=num_select, replace=False, p=probs))
    worst_idx = np.unique(np.concatenate(worst_idx))
    worst_idx.sort()

    selected = [atoms_filt[i] for i in worst_idx]
    if len(selected) > num_select:
        indices = np.random.choice(np.arange(len(selected)), size=num_select, replace=False)
        selected = [selected[i] for i in indices]

    output_path = workdir / "tps_selected_frames.xyz"
    write(output_path, selected)
    return output_path
