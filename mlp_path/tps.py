import json
import numpy as np 
import matplotlib.pyplot as plt
# import openpathsampling as paths
import torch
import sys
import os
from functools import reduce
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import argparse
import time
from pathlib import Path

import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage

# Monkey patch OpenPathSampling to use experimental features
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

import json
import MDAnalysis as mda

from util.parser import read_config, print_config
from multiprocessing import Process


import mdtraj

from util.parser import read_config, print_config, create_parser, global_arguments
from util import CV

from abc import ABC, abstractmethod

import pickle

from loguru import logger


# --- Interfaces / abstract base classes ---
class ISetup(ABC):
    def __init__(self, config_path, resource_directory="", print_config_bool=True):
        self.resource_directory = resource_directory
        self.config = read_config(config_path)
        if print_config_bool:
            print_config(self.config)

        #fixed units
        self.size_unit = getattr(unit, self.config['UNITS']['size_unit'])
        # if self.constant_variable == "P":
            # self.press_unit = getattr(unit, self.config['UNITS']['press_unit'])
        self.temp_unit = getattr(unit, self.config['UNITS']['temp_unit'])
        self.time_unit = getattr(unit, self.config['UNITS']['time_unit'])


        # Simulation settings from config
        integrator_config = self.config.get("Integrator_settings")
        self.temperature = integrator_config['temperature'] * self.temp_unit
        self.dt = integrator_config["dt"] * self.time_unit
        self.friction = integrator_config["friction"] / self.time_unit
        self.constant_variable = integrator_config["constant_variable"]
        if self.constant_variable == "P":
            self.barostat_frequency = integrator_config["barostat_frequency"]
            self.pressure = integrator_config['pressure'] * self.press_unit
        else:
            try:
                if integrator_config["barostat_frequency"]:
                    print("Warning: constant_variable is not 'P'. Ignoring barostat settings.") 
            except KeyError:
                pass
        #System name
        self.system_name = self.config['system_name']
        self.platform_name = self.config["platform_name"]
        
        #MACE 
        self.mace_path = self.config['mace_model_path']
        if isinstance(self.mace_path, list):
            self.mace_path = self.mace_path[0]  # take first for single model setup
        # TPS Specific settings
        self.TPS_settings = self.config.get("TPS_settings")
        self.n_frames_max = self.TPS_settings["n_frames_max"]
        self.n_steps_per_frame = self.TPS_settings["n_steps_per_frame"]

        # Initialize components
        self.integrator_temp = self._setup_integrator()
        self._topology, self.system, self.positions = self._setup_system()
        self.template = self._create_template()
        self.topology = self.template.topology
        self.integrator = self._setup_integrator()
        self.md_engine = self._setup_engine(self.integrator)

        # Define states
        
        # self.states = self._define_states(self.config.get("TPS_settings"))

    def _setup_integrator(self):
        print("Setting up integrator...")
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.dt
        )
        integrator.setConstraintTolerance(0.00001)
        return integrator

    def _setup_engine(self, integrator):
        print(f"Setting up engine on {self.platform_name} platform...")
        platform = mm.Platform.getPlatformByName(self.platform_name)
        engine_options = {
            'n_frames_max': self.n_frames_max,
            'n_steps_per_frame': self.n_steps_per_frame
        }
        engine = eng.Engine(
            topology=self.topology,
            system=self.system,
            integrator=integrator,
            openmm_properties=self._get_platform_properties(),
            options=engine_options
        )
        engine.name = 'default'
        engine.initialize(platform)
        
        return engine

    def _setup_system(self):
        
        # Obtain an initial state of the system 
        # with the correct corresponding boundaries.
        logger.debug("Extracting the topology and creating the system...")
        logger.debug(self.resource_directory)
        logger.debug(os.getcwd())
        atoms = read(self.config.get("top_file","top.pdb"))

        pos = atoms.get_positions() / 10
        box_vectors = atoms.get_cell() / 10

        elements = atoms.get_chemical_symbols()

        # Create a topology object
        topology = Topology()

        # Add atoms to the topology

        for i, (element, position) in enumerate(zip(elements, pos)):
            e = Element.getBySymbol(element)
            chain = topology.addChain(element)
            res = topology.addResidue(element, chain)
            topology.addAtom(element, e, res,id=element)
            # print(element )
        # if there is a periodic box specified add it to the Topology
        if max(atoms.get_cell().cellpar()[:3]) > 0:
            topology.setPeriodicBoxVectors(vectors=box_vectors)

        prmtop = MLPotential("mace",modelPath=self.mace_path,device="cuda")

        system = prmtop.createSystem(topology,nl="torch",    max_n_pairs=-1)


        

        if self.constant_variable == "P":
            system.addForce(mm.MonteCarloBarostat(self.pressure, self.temperature, self.barostat_frequency))
        
        init_simulation = app.Simulation(topology, system, self.integrator_temp)
        init_simulation.context.setPositions(pos)
        # init_simulation.loadState(str(self.resource_directory)+ "/cb7_b2_mdV_cont2.state")
        newboxVectors=  init_simulation.context.getState(getParameters=True).getPeriodicBoxVectors()
        init_simulation.topology.setPeriodicBoxVectors(newboxVectors)

        positions = init_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

        # print(positions)
        return topology, system, positions

    def _get_platform_properties(self):
        if self.platform_name == 'OpenCL':
            return {'OpenCLPrecision': 'mixed', 'OpenCLDeviceIndex': "2"}
        elif self.platform_name == 'CUDA':
            return {'CudaPrecision': 'mixed'}
        elif self.platform_name == 'CPU':
            return {}
        else:
            return {}

    def _create_template(self):
        print("Creating snapshot template...")
        testsystem = namedtuple('LocalTestSystem', ['name', 'system', 'topology', 'positions'])
        test_system = testsystem(name='CB7:B2', system=self.system, topology=self._topology, positions=self.positions)
        return eng.snapshot_from_testsystem(test_system)

class ISampler(ABC):
    @abstractmethod
    def run(self):
        pass



class Sampler(ISampler):
    def __init__(
        self,
        setup_class:ISetup,  # e.g. TPS_setup or TIS_setup
        directory=".",
        output_path=None,
        ops_storage_path="tps_storage.db",
        save_final_traj=False,
        final_traj_storage_path=None,
        system_resource_directory="",
        prepare_only=False,  # <--- new flag
        absolute_paths=False,  # <--- new option
        pickle_path=None,  # optional path to save pickled sampler
    ):
        # store args
        self.setup_class = setup_class
        self.directory = Path(directory)
        self.config_file = setup_class.config
        self.n_steps = self.config_file.get("tot_mc_steps", 100)
        self.step = self.config_file.get("save_every", 20)
        self.traj_path = Path(self.config_file.get("init_path", "tps.xyz"))
        self.output_path = Path(output_path) if output_path else self.directory
        if os.path.exists(self.output_path) is False:
            os.makedirs(self.output_path)
        self.ops_storage_path = ops_storage_path
        self.save_final_traj = save_final_traj
        self.final_traj_storage_path = final_traj_storage_path
        self.system_resource_directory = system_resource_directory 

        self.prepare_only = prepare_only
        self.absolute_paths = absolute_paths
        self.pickle_path = pickle_path

        # placeholders to be created later
        self.setup = None
        self.engine = None
        self.template = None

    def __getstate__(self):
        # Remove unpickleable objects
        state = self.__dict__.copy()
        # engine, setup, template etc should not be pickled
        for key in [ 'setup', 'engine', 'template']:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        # Restore state
        self.__dict__.update(state)
        # Recreate setup and engine when unpickled
        config_path = self.directory / self.config_file
        self.setup = self.setup_class(config_path, resource_directory=self.system_resource_directory)
        paths.PathMover.engine = self.setup.md_engine
        self.engine = self.setup.md_engine
        self.template = self.setup.template

    def run(self):
        # config_path = self.directory / self.config_file

        # --- HPC prepare-only mode ---
        if self.prepare_only:
            # create folders
            base_path = Path.cwd() if self.absolute_paths else Path(".")
            os.makedirs(self.output_path, exist_ok=True)
            # if self.final_traj_storage_path:
            #     os.makedirs(self.final_traj_storage_path, exist_ok=True)

            # save pickled sampler
            if self.pickle_path:
                with open(self.pickle_path, "wb") as f:
                    pickle.dump(self, f)
                logger.debug(f"[Sampler pickled to: {self.pickle_path}]")

            # print HPC command
            cmd = (
                f"python {Path(__file__).name} "
                f"--run_pickled {self.pickle_path}"
            )
            logger.debug(f"[Prepared workflow folder: {self.output_path}]")
            logger.debug("\nRun the following on HPC to execute the full TPS:\n")
            logger.debug(cmd)
            return cmd

        # --- If running from pickled object or normally ---
        # check if setup/engine already exists
        # if self.setup is None:
        #     setup = self.setup_class(config_path, resource_directory=self.system_resource_directory)
        #     paths.PathMover.engine = self.setup.md_engine
        #     self.engine = self.setup.md_engine
        #     self.template = self.setup.template

        
        # --- normal run mode ---
        setup = self.setup_class

        paths.PathMover.engine = setup.md_engine

        # load initial trajectory
        traj_path = self.traj_path
        if traj_path.suffix == ".nc":
            old_store = paths.Storage(traj_path)
            init_traj = old_store.tag["trajectory"]
            mdtraj_traj = init_traj.to_mdtraj()
            old_store.close()
            init_traj = ops_openmm.trajectory_from_mdtraj(mdtraj_traj)
        elif traj_path.suffix == ".db":
            old_store = Storage(traj_path)
            init_traj = old_store.trajectories[-1]
            mdtraj_traj = init_traj.to_mdtraj()
            old_store.close()
            init_traj = ops_openmm.trajectory_from_mdtraj(mdtraj_traj)
        elif traj_path.suffix == ".xyz":
            mdt = mdtraj.load(self.config_file.get("top_file","top.pdb"))
            print("top loaded")
            traj = mdtraj.load(traj_path, top=mdt)
            print("traj loaded",)
            traj.unitcell_vectors = np.array(list(mdt.unitcell_vectors) * len(traj.xyz))
            print("unit cell set",list(mdt.unitcell_vectors))
            init_traj = ops_openmm.trajectory_from_mdtraj(traj, velocities=[np.zeros_like(frame) for frame in traj.xyz])
            print("init traj created",init_traj)
        else:
            raise ValueError("Unsupported trajectory format")

        # create storage
        ops_storage_path = self.output_path / (self.ops_storage_path or f"d_tps_storage_{setup.system_name}.db")
        print(f"set {ops_storage_path}...")

        # --- build TPS/TIS sampler ---
        selector = paths.shooting.UniformSelector()
        beta = 1.0 / (setup.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
        modifier = paths.RandomVelocities(beta=beta, engine=setup.md_engine)
        tw_strategy = paths.strategies.TwoWayShootingStrategy(
            modifier=modifier, selector=selector, engine=setup.md_engine, group="TwoWayShooting"
        )

        # generic network setup placeholder (replace later for TIS)
        print("Defining CV and states...")
        cv = CV(self.config_file, self.traj_path)
        print("Creating TPS network...","states:",cv.states)
        network = paths.TPSNetwork.from_states_all_to_all(cv.states)

        move_scheme = paths.MoveScheme(network=network)
        move_scheme.append(tw_strategy)
        move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
        move_scheme.build_move_decision_tree()

        # initial conditions
        initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
        initial_conditions.sanity_check()

        # storage
        print("Creating storage...")
        storage = Storage(ops_storage_path, "w")
        for obj in [setup.template, setup.md_engine, setup.topology, initial_conditions, move_scheme]:
            storage.save(obj)

        sampler = paths.PathSampling(storage, move_scheme, initial_conditions).named("sampler")

        step = self.step
        os.makedirs(self.final_traj_storage_path, exist_ok=True)
        for i in range(0, self.n_steps + 1, step):
            sampler.run(step)
            for j, s in enumerate(storage.steps):
                if s.change.accepted:
                    s.active[0].trajectory.to_mdtraj().save(f"{self.final_traj_storage_path}/accepted_{j}.xyz")

        # final traj
        if self.save_final_traj:
            final_path = self.output_path / (self.final_traj_storage_path or f"tps_final_{setup.system_name}.db")
            final_storage = Storage(final_path, "w")
            final_storage.save(storage.steps[-1].active[0].trajectory)
            final_storage.close()
        storage.close()
    
    @staticmethod
    def path_extension(traj, n_steps, engine,direction=["forward","backward"]):
        """Extend a trajectory by n_steps using the given engine."""
        last_frame = traj.frames[-1]
        engine.current_snapshot = last_frame
        extended_traj = engine.generate_trajectory(n_steps=n_steps)
        return traj + extended_traj




class TPS_setup(ISetup):
    def __init__(self, config_path: Path, resource_directory: Path = Path("."), print_config_bool=True):
        super().__init__( config_path, resource_directory, print_config_bool)
        # logger.debug(self.config)

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--run_pickled", type=str, help="Path to pickled Sampler object to run")
parser.add_argument("--prepare_only", action="store_true", help="Only prepare folders and print HPC command")
parser.add_argument("--absolute_paths", action="store_true", help="Use absolute paths in HPC command")
args, _ = parser.parse_known_args()

if args.run_pickled:
    # Load the pickled sampler
    with open(args.run_pickled, "rb") as f:
        sampler = pickle.load(f)
    sampler.prepare_only = False
    sampler.absolute_paths = False  # HPC run uses the paths in the pickle
    sampler.run()


if __name__ == "__main__":
    # Example usage
    tps = TPS_setup("tps.json")
    sampler = Sampler(
        setup_class=tps,
        directory=".",
        output_path="output",
        ops_storage_path="tps_storage.db",
        final_traj_storage_path="traj",
        system_resource_directory=".",
        absolute_paths=True
    )
 
    # sampler.run()

    # Extend the tps paths
    logger.debug("Extending TPS trajectories...")
    import ase.units as units
    nsteps = sampler.config_file.get("extend_steps", 100)
    timestep = sampler.config_file.get("Integrator_settings")["dt"]
    temp = sampler.config_file.get("Integrator_settings")["temperature"]
    friction = sampler.config_file.get("Integrator_settings")["friction"]

    logger.debug(f"Using device: {torch.device}")
    #check if multiple models are provided

    from mace.calculators import MACECalculator
    mace_model_paths = sampler.config_file.get("mace_model_path")
    if isinstance(mace_model_paths, list):
        # committee_models = []
        committee_calcs = [MACECalculator(model_paths=[m], device=sampler.config_file.get("platform_name", "cuda").lower()) for m in mace_model_paths]

        logger.debug(f"Using committee of {len(committee_calcs)} models for extension.")
    else:
        raise ValueError("mace_model_path should be a list of model paths for committee evaluation.")

    folder = Path(sampler.final_traj_storage_path).resolve()
    traj_list = [f for f in os.listdir(folder)
                if f.endswith(".xyz") and not f.startswith("._") and f != "accepted_0.xyz"]
    traj_list = [folder / f for f in traj_list]
    logger.info(f"Found {len(traj_list)} trajectories in {folder}.")

    from mlp import MACE_MLP
    evaluate_committee = MACE_MLP.evaluate_committee
    cell = read(Path(sampler.config_file.get("top_file","top.pdb")).resolve()).get_cell() 
    # make extend folder
    # os.makedirs(folder/"extend", exist_ok=True)
    extend_folder =     folder/"extend"
    os.makedirs(extend_folder, exist_ok=True)

    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.langevin import Langevin


    for traj_file in traj_list:

        logger.info(f"Processing {traj_file}")
        base_name = traj_file.name.replace(".xyz", "")

        # --- read only first and last frames efficiently ---
        logger.info(f"  Reading trajectory.. {traj_file}")
        orig_traj = read(f"{traj_file}", index=":")
        atoms_first = orig_traj[0].copy()
        atoms_last = orig_traj[-1].copy()

        frames = []
        for i, atoms in enumerate(orig_traj):
            atoms = atoms.copy()
            atoms.cell = cell
            # atoms.center()
            atoms.set_pbc((True, True, True))
            atoms.calc = committee_calcs[0]
            # if i % 100 == 0:
            evaluate_committee(atoms,committee_calcs=committee_calcs)
            frames.append(atoms.copy())
            logger.info(f"    Frame {i}/{len(orig_traj)} processed")

        out_file = f"{base_name}.xyz"
        write(extend_folder/out_file, frames)
        logger.info(f"  Saved full trajectory with committee stats → {out_file}")
        frames.clear()

        for direction, atoms in zip(["backward", "forward"], [atoms_first, atoms_last]):
            logger.info(f"  Running {direction} MD")

            atoms.cell = cell
            # atoms.center()
            atoms.set_pbc((True, True, True))
            atoms.calc = committee_calcs[0]

            # initialize random velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
            if direction == "backward":
                atoms.set_velocities(-atoms.get_velocities())

            out_file = f"{base_name}_{direction}.xyz"
            if os.path.exists(extend_folder/out_file):
                os.remove(extend_folder/out_file)

            frames = []  # buffer

            dyn = Langevin(atoms, timestep, temperature_K=temp, friction=friction)
            def buffer_frame():
                logger.info(f"    Step {dyn.nsteps}/{nsteps} ")
                # c_time = time.time()
                # logger.info(f"Time per frame {c_time - start:.1f}s")
                evaluate_committee(atoms,committee_calcs=committee_calcs)
                frames.append(atoms.copy())
                
            # def write_frame():
                # evaluate_committee(atoms)
                # atoms.write(out_file, append=True, write_results=False)

            dyn.attach(buffer_frame, interval=1)

            start = time.time()
            dyn.run(nsteps)
            logger.info(f"    Runtime {direction}: {time.time() - start:.1f}s")
            for atoms in frames:
                # atoms.arrays['forces'] = atoms.get_forces()
                if 'momenta' in atoms.arrays:
                    del atoms.arrays['momenta']

            # write once at end
            write(extend_folder/out_file, frames)
            frames.clear()

            logger.info(f"    Saved → {out_file}")
            # --- end of MD ---


        # --- merge trajectories ---
        logger.info("  Merging extended trajectory...")
        backward = read(extend_folder/f"{base_name}_backward.xyz", index=":")
        forward = read(extend_folder/f"{base_name}_forward.xyz", index=":")
        backward = backward[::-1]  # reverse time order

        orig_traj = read(extend_folder/f"{base_name}.xyz", index=":")

        # orig_traj = read(f"{folder}/{traj}", index=":")
        extended = backward + orig_traj + forward
        out_name = f"{base_name}_extended.xyz"
        write(extend_folder/out_name, extended)
        logger.info(f"  Saved → {out_name}\n")
        