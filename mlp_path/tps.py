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
        logger.debug(f"Using units: size={self.size_unit}, temp={self.temp_unit}, time={self.time_unit}")


        # Simulation settings from config
        integrator_config = self.config.get("Integrator_settings")
        self.temperature = integrator_config['temperature'] * self.temp_unit
        self.dt = integrator_config["dt"] * self.time_unit
        self.friction = integrator_config["friction"] / self.time_unit
        logger.debug(f"Integrator settings: temperature={self.temperature}, dt={self.dt}, friction={self.friction}")
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
        self.step = self.config_file.get("save_every", None)
        if self.step is None:
            logger.warning(f"save_every not specified in config, you are doing only one save at the end of the {self.n_steps} TPS run.")
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

        # # --- HPC prepare-only mode ---
        # if self.prepare_only:
        #     # create folders
        #     base_path = Path.cwd() if self.absolute_paths else Path(".")
        #     os.makedirs(self.output_path, exist_ok=True)
        #     # if self.final_traj_storage_path:
        #     #     os.makedirs(self.final_traj_storage_path, exist_ok=True)

        #     # save pickled sampler
        #     if self.pickle_path:
        #         with open(self.pickle_path, "wb") as f:
        #             pickle.dump(self, f)
        #         logger.debug(f"[Sampler pickled to: {self.pickle_path}]")

        #     # print HPC command
        #     cmd = (
        #         f"python {Path(__file__).name} "
        #         f"--run_pickled {self.pickle_path}"
        #     )
        #     logger.debug(f"[Prepared workflow folder: {self.output_path}]")
        #     logger.debug("\nRun the following on HPC to execute the full TPS:\n")
        #     logger.debug(cmd)
        #     return cmd

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
        print(f" Time step {setup.md_engine.simulation.integrator.getStepSize()}")
        print(f" Time step {setup.integrator.getStepSize()}")


        # --- build TPS/TIS sampler ---
        selector = paths.shooting.UniformSelector()
        beta = 1.0 / (setup.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
        modifier = paths.RandomVelocities(beta=beta, engine=setup.md_engine)
        tw_strategy = paths.strategies.TwoWayShootingStrategy(
            modifier=modifier, selector=selector, engine=setup.md_engine, group="TwoWayShooting"
        )

        # generic network setup placeholder (replace later for TIS)
        logger.debug("Defining CV and states...")
        cv = CV(self.config_file, self.traj_path)
        logger.debug(f"Creating TPS network with states: {cv.states}")
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
        # for obj in [setup.template, setup.md_engine, setup.topology, initial_conditions, move_scheme]:
        #     storage.save(obj)
        # for obj in [setup.template, setup.topology, initial_conditions, move_scheme]:
        #     storage.save(obj)

        sampler = paths.PathSampling(storage, move_scheme, initial_conditions).named("sampler")

        save_rejected = self.config_file.get("save_rejected", False)

        print(f" Time step {setup.md_engine.simulation.integrator.getStepSize()} \n \n \n \n")
        if self.step is not None:
            step = self.step
            os.makedirs(self.final_traj_storage_path, exist_ok=True)
            for i in range(0, self.n_steps + 1, step):
                logger.debug(f"Running TPS steps {i} to {i + step}...")
                sampler.run(step)
                for (j, s), store in zip(enumerate(storage.steps), storage.trajectories):
                    if j <=i:
                        continue
                        
                    if s.change.accepted:
                        logger.debug(f"  Saving accepted trajectory for step {j}...")
                        storage.trajectories[j].to_mdtraj().save(f"{self.final_traj_storage_path}/accepted_{j}.xyz")
                    else:
                        if save_rejected:
                            logger.debug(f"  Saving rejected trajectory for step {j}...")
                            storage.trajectories[j].to_mdtraj().save(f"{self.final_traj_storage_path}/rejected_{j}.xyz")
            logger.debug("Saving initial Path.")
        else:
            logger.debug(f"Running full TPS for {self.n_steps} steps...")
            sampler.run(self.n_steps)
            logger.debug("Saving initial Path.")
        try:
            storage.steps[0].active[0].trajectory.to_mdtraj().save(f"{self.final_traj_storage_path}/accepted_0.xyz")
        except Exception as e:
            logger.error(f"Failed to save initial path: {e}")

        # final traj
        if self.save_final_traj:
            final_path = self.output_path / (self.final_traj_storage_path or f"tps_final_{setup.system_name}.db")
            final_storage = Storage(final_path, "w")
            final_storage.save(storage.steps[-1].active[0].trajectory)
            final_storage.close()
        storage.close()

    def save_from_storage(self, storage_path, out_folder):
        """Extract accepted and rejected trajectories from storage."""
        os.makedirs(out_folder, exist_ok=True)
        # make an empty engine to use for conversion
        setup = self.setup_class
        paths.PathMover.engine = setup.md_engine
        logger.debug(f"Loading storage from {storage_path}...")
        storage = Storage(storage_path, "r")
        storage.engine = None
        logger.debug(f"Saving trajectories to {out_folder}...")
        save_rejected = self.config_file.get("save_rejected", False)

        # reomve engine from storage 
        # storage.engine = None
        # steps = storage.steps
        # trajs = storage.trajectories
        end = len(storage.steps)
        i = end
        for s,traj  in zip(storage.steps[::-1],storage.trajectories[::-1]):
            if Path(f"{self.final_traj_storage_path}/accepted_{i}.xyz").exists() or Path(f"{self.final_traj_storage_path}/rejected_{i}.xyz").exists():
                logger.debug(f"  Trajectory for step {i} already exists, skipping...")
                i -= 1
                break
            if s.change.accepted:
                logger.debug(f"  Accepted trajectory for step {i}...")
                traj.to_mdtraj().save(f"{self.final_traj_storage_path}/accepted_{i}.xyz")
                logger.debug(f"Saved accepted trajectory for step {i}.")
            else:
                logger.debug(f" Rejected trajectory for step {i}...")
                if save_rejected:
                    traj.to_mdtraj().save(f"{self.final_traj_storage_path}/rejected_{i}.xyz")
                    logger.debug(f"Saved trajectory for step {i}.")     
            i -= 1
        storage.close()
        logger.debug("All trajectories saved.")
            
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




if __name__ == "__main__":
    # Example usage
    # load json config
    
    #take command line args for sampler run or read from storage and the json file name
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--run_pickled", type=str, help="Path to pickled Sampler object to run")
    # parser.add_argument("--prepare_only", action="store_true", help="Only prepare folders and print HPC command")
    # parser.add_argument("--absolute_paths", action="store_true", help="Use absolute paths in HPC command")
    # parser.add_argument("--db_path", type=str, help="Path to save pickled Sampler object")
    # argument for load from db or run sampler
    # parser.add_argument("--load_from_db", action="store_true", help="Load sampler from database instead of running",default=False)
    parser.add_argument("--start_type", type=str, choices=["new","from_db","continue"], default="new", help="Start a new TPS run, load from database, or extend and continue from saved trajectories")
    args, _ = parser.parse_known_args()



    # with open("tps.json", "r") as f:
    #     config = json.load(f)
    
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
    
    

    if args.start_type == "from_db":
        logger.debug(f"Saving trajectories from storage at {sampler.output_path / sampler.ops_storage_path} to {sampler.final_traj_storage_path}.")
        sampler.save_from_storage(sampler.output_path / sampler.ops_storage_path, sampler.final_traj_storage_path)
    elif args.start_type == "continue":
        logger.debug(f"Continuing TPS run by extending trajectories from {sampler.final_traj_storage_path}.")
    elif args.start_type == "new":
        logger.debug("Starting new TPS run.")
        sampler.run()
    else:
        raise ValueError("Invalid start_type argument.")


    # Extend the tps paths
    logger.debug("Extending TPS trajectories...")
    import ase.units as units
    nsteps = sampler.config_file.get("extend_steps", None)

    timestep = sampler.config_file.get("Integrator_settings")["dt"]*1000 #ps to fs
    temp = sampler.config_file.get("Integrator_settings")["temperature"]
    friction = sampler.config_file.get("Integrator_settings")["friction"]/1000 # 1/ps to 1/fs

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
                if f.endswith(".xyz") and not f.startswith("._")]
    traj_list = [folder / f for f in traj_list]

    #sort traj list
    traj_list = sorted(traj_list, key=lambda x: int(x.stem.split("_")[-1].split(".")[0]))

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

        if os.path.exists(extend_folder/f"{base_name}_extended.xyz"):
            logger.info(f"  Extended trajectory already exists, skipping...")
            continue

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
            # logger.info(f"    Frame {i}/{len(orig_traj)} processed")

        out_file = f"{base_name}.xyz"
        write(extend_folder/out_file, frames)
        logger.info(f"  Saved full trajectory with committee stats → {out_file}")
        frames.clear()
        if "accepted" in base_name:
            if nsteps:
                # --- extend both ends ---
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

                    dyn = Langevin(atoms, timestep * units.fs, temperature_K=temp, friction=friction)
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
            else:
                # just copy original to extended
                out_name = f"{base_name}_extended.xyz"
                orig_traj = read(extend_folder/f"{base_name}.xyz", index=":")
                write(extend_folder/out_name, orig_traj)
                logger.info(f"  No extension requested, copied original to → {out_name}\n")



    #combine all extended trajs into one
    logger.info("Combining all extended trajectories into one...")
    combined = []
    extend_traj_list = [f for f in os.listdir(extend_folder)
                if (f.endswith("_extended.xyz") or f.startswith("rejected")) and not f.startswith("._") ]
    extend_traj_list = [extend_folder / f for f in extend_traj_list]
    extend_traj_list = sorted(extend_traj_list, key=lambda x: int(x.stem.strip("_extended.xyz").split("_")[-1]))

    for traj_file in extend_traj_list:
        logger.info(f"  Reading {traj_file}")
        traj = read(traj_file, index=":")
        combined.extend(traj)
    combined_out = sampler.output_path / "tps_all_extended.xyz"
    write(combined_out, combined)
    logger.info(f"Saved combined extended trajectory → {combined_out}")

    from util.mlp_helper import process_frame
    from ase.io import read, write
    import numpy as np

    # --- analyze extended traj for high-variance frames ---
    logger.info("Analyzing extended trajectory for high-variance frames...")

    cutoff = sampler.config_file.get("local_cutoff", 5.0)
    carbon_indices = sampler.config_file.get("selection_indices")
    num_select = sampler.config_file.get("num_select")

    # new thresholds
    e_thresh = sampler.config_file.get("e_var_max", 1000)
    f_thresh = sampler.config_file.get("f_var_max", 100)

    atoms_list = read(sampler.output_path / "tps_all_extended.xyz", index=":")

    # compute metrics
    metrics = []
    for at in atoms_list:
        e_var, f_var, local_mean, local_max = process_frame(at, carbon_indices, cutoff)
        metrics.append((e_var, f_var, local_mean, local_max))
    metrics = np.array(metrics)

    logger.debug(f"metrics computed: {metrics}")

    # --- apply thresholds ---
    mask = (metrics[:, 0] <= e_thresh) & (metrics[:, 1] <= f_thresh)

    metrics_filt = metrics[mask]
    atoms_filt = [a for a, keep in zip(atoms_list, mask) if keep]

    if len(metrics_filt) == 0:
        logger.warning("No frames passed threshold filtering.")
        metrics_filt = metrics
        atoms_filt = atoms_list

    # --- take worst num_select from each column ---
    worst_idx = []
    alpha = sampler.config_file.get("alpha", 0)

    for col in range(metrics_filt.shape[1]):
        vals = metrics_filt[:, col]

        # rescale to 0–1 (keeps relative differences even if tiny)
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)

        # mix
        weights = alpha * vals_norm + (1 - alpha)

        # probabilities
        probs = weights / weights.sum()

        topk = np.random.choice(len(vals), size=num_select, replace=False, p=probs)
        worst_idx.append(topk)

    # combine and dedupe
    worst_idx = np.unique(np.concatenate(worst_idx))
    worst_idx.sort()

    # final selected frames
    selected = [atoms_filt[i] for i in worst_idx]

    #randomly select num_select frames if not enough to make sure its num_select
    if len(selected) > num_select:
        indices = np.arange(len(selected))
        selected_indices = np.random.choice(indices, size=num_select, replace=False).tolist()
        selected = [selected[i] for i in selected_indices]

    output_path = sampler.output_path / "tps_selected_frames.xyz"
    write(output_path, selected)



        # write out

    