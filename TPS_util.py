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



class TPS_setup:
    def __init__(self, config_path, resource_directory="", print_config_bool=True):
        self.resource_directory = resource_directory
        self.config = read_config(config_path)
        if print_config_bool:
            print_config(self.config)

        #fixed units
        self.size_unit = getattr(unit, self.config['UNITS']['size_unit'])
        # self.press_unit = getattr(unit, self.config['UNITS']['press_unit'])
        self.temp_unit = getattr(unit, self.config['UNITS']['temp_unit'])
        self.time_unit = getattr(unit, self.config['UNITS']['time_unit'])


        # Simulation settings from config
        integrator_config = self.config.get("Integrator_settings")
        # self.pressure = integrator_config['pressure'] * self.press_unit
        self.temperature = integrator_config['temperature'] * self.temp_unit
        self.dt = integrator_config["dt"] * self.time_unit
        self.friction = integrator_config["friction"] / self.time_unit
        # self.barostat_frequency = integrator_config["barostat_frequency"]
        self.constant_variable = integrator_config["constant_variable"]
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
        # CHANGE
        # Obtain an initial state of the system 
        # with the correct corresponding boundaries.
        print("Extracting the topology and creating the system...")
        print(self.resource_directory)
        print(os.path.dirname(os.path.abspath(os.getcwd())))
        atoms = read('top.pdb')

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


        # )

        if self.constant_variable == "P":
            system.addForce(mm.MonteCarloBarostat(self.pressure, self.temperature, self.barostat_frequency))
        
        init_simulation = app.Simulation(topology, system, self.integrator_temp)
        init_simulation.context.setPositions(pos)
        # init_simulation.loadState(str(self.resource_directory)+ "/cb7_b2_mdV_cont2.state")
        newboxVectors=  init_simulation.context.getState(getParameters=True).getPeriodicBoxVectors()
        init_simulation.topology.setPeriodicBoxVectors(newboxVectors)
        # print("hier ben ik nu 2")
        # print(newboxVectors)
        oldboxVectors = init_simulation.topology.getPeriodicBoxVectors()
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

        return (bound, unbound)



def create_parser(arguments_list):
    """
    Creates an ArgumentParser with arguments specified in arguments_list.

    Args:
        arguments_list (list): List of tuples or dictionaries defining arguments:
            - If tuple: (flags, type, required, help_text, [default], [nargs])
            - If dictionary: {
                "flags": list of flags (e.g., ["-d", "--directory"]),
                "type": argument type (e.g., str, int, Path),
                "required": if the argument is required,
                "help": help message for the argument,
                "default": (optional) default value if argument is not provided,
                "nargs": (optional) number of arguments expected (e.g., "+", "*")
              }

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser
    """
    parser = argparse.ArgumentParser()
    for arg in arguments_list:
        if isinstance(arg, tuple):  # Handle tuple format
            # Unpack flags, type, required, help_text, with optional default and nargs
            flags, arg_type, required, help_text, *rest = arg
            default = rest[0] if len(rest) > 0 and not isinstance(rest[0], str) else None
            nargs = rest[1] if len(rest) > 1 and isinstance(rest[1], str) else None
            
            # Add the argument to the parser
            parser.add_argument(
                *flags,
                type=arg_type,
                required=required,
                help=help_text,
                default=default,
                nargs=nargs
            )
        elif isinstance(arg, dict):  # Handle dictionary format
            # Add the argument to the parser using dict keys
            parser.add_argument(
                *arg["flags"],
                type=arg["type"],
                required=arg.get("required", False),
                help=arg.get("help", ""),
                default=arg.get("default", None),
                nargs=arg.get("nargs", None)
            )
    return parser


def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError as e:
            raise(ValueError(f"Error parsing JSON configuration file: {e}"))
    return config

def print_config(config, prefix=''):
    if isinstance(config, dict):
        for key, value in config.items():
            new_prefix = f"{prefix}{key}: "
            if isinstance(value, (dict, list)):
                print(new_prefix)
                print_config(value, prefix + '    ')
            else:
                print(f"{new_prefix}{value}")
    elif isinstance(config, list):
        for index, item in enumerate(config):
            new_prefix = f"{prefix}[{index}]: "
            if isinstance(item, (dict, list)):
                print(new_prefix)
                print_config(item, prefix + '    ')
            else:
                print(f"{new_prefix}{item}")


# Global argument dictionary
global_arguments = {
    "directory": (["-dir", "--directory"], Path, True, "Directory containing initial trajectory and configuration files."),
    "config_file": (["-cfg", "--config_file"], str, True, "File in python configparser format with simulation settings."),
    "n_steps": (["-nr", "--n_steps"], int, True, "The number of desired TPS/TIS MC cycles."),
    "output_path": (["-out", "--output_path"], Path, False, "Directory for storing TPS output files."),
    "previous_model": (["-previous_model", "--previous_model"], str, False, "Name of the old AIMMD model to be used."),
    "aimmd_store": (["-aimmd_store", "--aimmd_store"], str, False, "Name for output AIMMD storage, default describes which selector and PES is used."),
    "ops_store": (["-ops_store", "--ops_store"], str, False, "Name for output OPS storage, default describes which selector and PES is used."),
    "final_storage": (["-final_storage", "--final_storage"], str, False, "Save final trajectory as separate OPS storage."),
    "trajectory_path": (["-traj", "--trajectory_path"], Path, False, "Path to the input trajectory storage file."),
    "save_final_traj": (["-save_traj", "--save_final_traj"], bool, False, "Flag to save the final trajectory as a separate OPS storage.", True),
    "final_traj_storage_path": (["-final_traj_store", "--final_traj_storage_path"], str, False, "Path to save the final trajectory storage file."),
    "system_resource_directory": (["-sys_source_dir", "--system_resource_directory"], Path, False, "Directory for system resource files."),
    "n_epoch": (["-n_epoch", "--n_epoch"], int, True, "The number of epochs for training."),
    "stable_ops_storages":(["-stable_ops_store", "--stable_ops_store"], str, False, "Paths to OPS storage files for stable states A and B.", None, "+"),
    "config_file_TPS": (["-cfg_TPS", "--config_file_TPS"], str, True, "File in python configparser format with simulation settings."),
    "config_file_TIS": (["-cfg_TIS", "--config_file_TIS"], str, True, "File in python configparser format with TIS settings."),
    "interface_model": (["-interface_model", "--interface_model"], str, True, "Name of the old AIMMD model to be used."),
    "interface_values": (["-interface_values", "--interface_values"], float, True, "The value of the interface used in the simulation.", None, "+"),
    "direction": (["-direction", "--direction"], str, True, "The direction of the simulation, either 'forward' or 'backward'.", None, None, ["forward", "backward"]),
    "RPE_name": (["-RPE_name","--RPE_name"],str,False,"RPE storage name, default name is RPE_storage"),
    "number_trajectories": (["-n_traj", "--n_traj"], int, False, "The number of trajectories obtained for each TIS run."),
    "n_thermalization": (["-n_therm", "--n_therm"], int, False, "The number of trajectories removed as thermalization.", 200),
    "cutoff": (["-cutoff", "--cutoff"], float, False, "The WHAM cutoff used.", 0.01),
    "max_min_filename": (["-max_min", "--max_min"], str, False, "filename for the max_min stored path (should be first part of the name e.g. 'max_min_q_int')")
}


