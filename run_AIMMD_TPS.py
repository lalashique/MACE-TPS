import numpy as np
import matplotlib.pyplot as plt
import openpathsampling as paths
import argparse
from pathlib import Path
import mdtraj
from simtk import unit
from multiprocessing import Process
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage

# Monkey patch OpenPathSampling to use experimental features
paths = monkey_patch_all(paths)

from TPS_util import TPS_setup, create_parser, global_arguments
from AIMMD_utils import AIMMD_setup
from CV import CV

import MDAnalysis as mda
from ase.io import read, write
import os
import aimmd
import torch

# Define the main function to run AIMMD TPS
def run_AIMMD_TPS(input_path: Path = None, config_path=None, n_steps=None, traj_path=None, output_path=None, 
                  ops_storage_path=None, save_final_traj=True, final_traj_storage_path=None, system_resource_directory=None,    
                  previous_model_file=None, aimmd_store_path=None
                  ):
    print("torch.cuda.is_available():", torch.cuda.is_available())
    # Construct the full config path
    if system_resource_directory is None:
        system_resource_directory= ""
    config_path = Path(input_path / config_path).with_suffix(".json")
    TPS_utils = TPS_setup(config_path, resource_directory=system_resource_directory, print_config_bool=True)
    paths.PathMover.engine = TPS_utils.md_engine
    
    # Load or initialize initial trajectory
    if traj_path:
        # Check the file extension
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

            mdt = mdtraj.load('top.pdb')
            traj = mdtraj.load(traj_path,top=mdt)
            traj.unitcell_vectors = np.array(list(mdt.unitcell_vectors)*len(traj.xyz))
            init_traj = ops_openmm.trajectory_from_mdtraj(traj,velocities=[np.zeros_like(frame) for frame in traj.xyz])

        else:
            raise ValueError("Unsupported file type. Use .nc or .db for trajectory storage.")
    else:
        raise ValueError("traj_path is required to load initial trajectory.")
    # print(init_traj.to_dict())
    # print(init_traj.objects())

    # Create storage paths
    if ops_storage_path is None:
        ops_storage_path = "d_tps_storage_{}".format(TPS_utils.system_name)
    ops_storage_path = Path(output_path / ops_storage_path).with_suffix(".db")
    
    # Transition network to sample transitions
    cv = CV(config_path, traj_path)

    descriptor_transform = cv.descriptor_transform

    AIMMD_set = AIMMD_setup(config_path, len(descriptor_transform(TPS_utils.template)), cv.states, descriptor_transform=descriptor_transform)

    if aimmd_store_path is None:
        aimmd_store_path = "aimmd_d_tps_storage_{}".format(AIMMD_set.distribution, TPS_utils.system_name)
    aimmd_store_path = Path(output_path / aimmd_store_path).with_suffix(".h5")

    if previous_model_file is not None:
        previous_model_file = Path(input_path / previous_model_file).with_suffix(".h5")

    # Create AIMMD storage
    aimmd_store = aimmd.Storage(aimmd_store_path, "w")
    # Create the committor model 
    model = AIMMD_set.setup_RCModel(aimmd_store, load_model_path=previous_model_file)

    # Initialize AIMMD hooks
    trainset = aimmd.TrainSet(n_states=2)
    trainhook = aimmd.ops.TrainingHook(model, trainset)
    storehook = aimmd.ops.AimmdStorageHook(aimmd_store, model, trainset)
    densityhook = aimmd.ops.DensityCollectionHook(model)


    # Setup the selector
    selector = AIMMD_set.setup_selector(model)

    # Velocity randomizer setup
    beta =  1. / (TPS_utils.integrator.getTemperature()* unit.BOLTZMANN_CONSTANT_kB)
    modifier = paths.RandomVelocities(beta=beta, engine=TPS_utils.md_engine)
    
    # Shooting strategy: TwoWayShooting
    tw_strategy = paths.strategies.TwoWayShootingStrategy(modifier=modifier, selector=selector, engine=TPS_utils.md_engine, group='TwoWayShooting')
    

    network = paths.TPSNetwork.from_states_all_to_all(cv.states)

    # Move scheme for sampling
    move_scheme = paths.MoveScheme(network=network)
    move_scheme.append(tw_strategy)
    move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
    move_scheme.build_move_decision_tree()

    # Initial conditions

    initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
    initial_conditions.sanity_check()

    # Create storage
    storage = Storage(ops_storage_path, "w")
    storage.save(TPS_utils.template)
    storage.save(TPS_utils.md_engine)

    storage.save(TPS_utils.topology)
    # storage.save(init_traj)
    storage.save(initial_conditions)
    storage.save(move_scheme)

    print(initial_conditions)
    

    # Create sampler and attach AIMMD components
    sampler = paths.PathSampling(storage, move_scheme, initial_conditions).named("TPS_sampler")
    sampler.attach_hook(trainhook)
    sampler.attach_hook(storehook)
    sampler.attach_hook(densityhook)
   
    step = 20 
    # mkdir output directory for accepted trajectories
    if not os.path.exists("traj"):
        os.makedirs("traj")
    for i in range(0,n_steps+1,step):
    # Run the sampler
        sampler.run(step)
        for i in range(len(storage.steps)):
            #TODO this now overrirdes previous files, need to fix and thuys has an additional overhead of saving all accepted trajs again
            if storage.steps[i].change.accepted:
                storage.steps[i].active[0].trajectory.to_mdtraj().save(f"traj/accepted_{i}.xyz")
    
    # Store the final trajectory in a separate file
    if save_final_traj:
        if final_traj_storage_path is None:
            final_traj_storage_path = "tps_storage_final_traj{}.db".format(TPS_utils.system_name)
        final_traj_storage_path = Path(output_path / final_traj_storage_path).with_suffix(".db")
        final_storage = Storage(final_traj_storage_path, "w")
        final_traj = storage.steps[-1].active[0].trajectory
        final_storage.save(final_traj)
        final_storage.close()
    
    # Save trainset and close storages
    storage.close()

# Main entry point for the script
if __name__ == "__main__":
    # Parse command-line arguments

    arguments_list = [
        global_arguments["directory"],
        global_arguments["config_file"],
        global_arguments["n_steps"],
        global_arguments["trajectory_path"],
        global_arguments["output_path"],
        global_arguments["ops_store"],
        global_arguments["final_storage"],
        global_arguments["system_resource_directory"],
        global_arguments["previous_model"],
        global_arguments["aimmd_store"]
    ]
    # Generate the parser
    parser = create_parser(arguments_list)
    args = parser.parse_args()
    
    # Retrieve command-line argument values
    in_path = args.directory
    configfile = args.config_file
    nsteps = args.n_steps
    outpath = args.output_path
    traj_path = args.trajectory_path
    ops_storage_path = args.ops_store 
    final_traj_storage_path = args.final_storage
    system_resource_directory = args.system_resource_directory
    previous_model_file = args.previous_model
    aimmd_store_path = args.aimmd_store
    
    # Determine whether to save the final trajectory
    save_final_traj = False
    if final_traj_storage_path is not None:
        save_final_traj = True
    if outpath is None:
        outpath = in_path
    # Run the process with the specified arguments
    process = Process(target=run_AIMMD_TPS, args=(in_path, configfile, nsteps, traj_path,  outpath, 
                                                  ops_storage_path, save_final_traj, final_traj_storage_path, 
                                                  system_resource_directory, previous_model_file, aimmd_store_path))
    process.start()
