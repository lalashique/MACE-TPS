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
from pathlib import Path

from openmm import app
import openmm as mm
from openmmtools.integrators import VVVRIntegrator
from simtk import unit
import openpathsampling as paths
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

import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage

import mdtraj

from util.parser import read_config, print_config, create_parser, global_arguments
from util import CV

from abc import ABC, abstractmethod

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
        print("Extracting the topology and creating the system...")
        print(self.resource_directory)
        print(os.path.dirname(os.path.abspath(os.getcwd())))
        atoms = read('../top.pdb')

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

        prmtop = MLPotential("mace",modelPath=self.mace_path)

        system = prmtop.createSystem(topology,nl="torch",    max_n_pairs=-1)


        

        if self.constant_variable == "P":
            system.addForce(mm.MonteCarloBarostat(self.pressure, self.temperature, self.barostat_frequency))
        
        init_simulation = app.Simulation(topology, system, self.integrator_temp)
        init_simulation.context.setPositions(pos)
        # init_simulation.loadState(str(self.resource_directory)+ "/cb7_b2_mdV_cont2.state")
        newboxVectors=  init_simulation.context.getState(getParameters=True).getPeriodicBoxVectors()
        init_simulation.topology.setPeriodicBoxVectors(newboxVectors)
        # print("hier ben ik nu 2")
        # print(newboxVectors)
        # oldboxVectors = init_simulation.topology.getPeriodicBoxVectors()
        # print(oldboxVectors)
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
        config_file="config.json",
        n_steps=1000,
        step=20,
        trajectory_path="p1.xyz",
        output_path=None,
        ops_storage_path=None,
        save_final_traj=False,
        final_traj_storage_path=None,
        system_resource_directory="",
    ):
        # store args
        self.setup_class = setup_class
        self.directory = Path(directory)
        self.config_file = config_file
        self.n_steps = n_steps
        self.step = step
        self.traj_path = Path(trajectory_path)
        self.output_path = Path(output_path) if output_path else self.directory
        self.ops_storage_path = ops_storage_path
        self.save_final_traj = save_final_traj
        self.final_traj_storage_path = final_traj_storage_path
        self.system_resource_directory = system_resource_directory 


    def run(self):
        config_path = self.directory / self.config_file
        setup = self.setup_class(config_path, resource_directory=self.system_resource_directory, print_config_bool=True)

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
            mdt = mdtraj.load("../top.pdb")
            traj = mdtraj.load(traj_path, top=mdt)
            traj.unitcell_vectors = np.array(list(mdt.unitcell_vectors) * len(traj.xyz))
            init_traj = ops_openmm.trajectory_from_mdtraj(traj, velocities=[np.zeros_like(frame) for frame in traj.xyz])
        else:
            raise ValueError("Unsupported trajectory format")

        # create storage
        ops_storage_path = self.output_path / (self.ops_storage_path or f"d_tps_storage_{setup.system_name}.db")

        # --- build TPS/TIS sampler ---
        selector = paths.shooting.UniformSelector()
        beta = 1.0 / (setup.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
        modifier = paths.RandomVelocities(beta=beta, engine=setup.md_engine)
        tw_strategy = paths.strategies.TwoWayShootingStrategy(
            modifier=modifier, selector=selector, engine=setup.md_engine, group="TwoWayShooting"
        )

        # generic network setup placeholder (replace later for TIS)
        cv = CV(config_path, self.traj_path)
        network = paths.TPSNetwork.from_states_all_to_all(cv.states)

        move_scheme = paths.MoveScheme(network=network)
        move_scheme.append(tw_strategy)
        move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
        move_scheme.build_move_decision_tree()

        # initial conditions
        initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
        initial_conditions.sanity_check()

        # storage
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




class TPS_setup(ISetup):
    def __init__(self, config_path: Path, resource_directory: Path = Path("."), print_config_bool=True):
        super().__init__(config_path, resource_directory, print_config_bool)


if __name__ == "__main__":
    # Example usage
    tps = TPS_setup("../config.json")
    sampler = Sampler(
        setup_class=TPS_setup,
        directory=".",
        config_file="../config.json",
        n_steps=100,
        trajectory_path="../p1.xyz",
        output_path="../output",
        ops_storage_path="../tps_storage.db",
        save_final_traj=True,
        final_traj_storage_path="../output/final_traj",
        system_resource_directory="../"
    )
    sampler.run()   