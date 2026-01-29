import tps
import os 
import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage

from ase import Atoms
from ase.io import write,read

# Monkey patch OpenPathSampling to use experimental features
paths = monkey_patch_all(paths)

from loguru import logger
from pathlib import Path
import argparse

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import numpy as np

def save_from_storage( storage_path, config, out_folder):
    """Extract accepted and rejected trajectories from storage."""
    def save_xyz(trajectory,file_name,type="mdtraj",topology=None):
        if type == "mdtraj":
            trajectory.to_mdtraj().save(file_name)
        elif type == "ase":
            if topology is None:
                raise ValueError("Topology must be provided for ASE format.")
            traj_ase = []
            for snap in trajectory:
                pos = snap.xyz*10  # nm to angstrom
                symbols = [atom.element.symbol if atom.element else atom.name[0] for atom in topology.atoms]
                atoms = Atoms(positions=pos, symbols=symbols)
                traj_ase.append(atoms)
            write(file_name, traj_ase)
        else:
            raise ValueError(f"Unsupported trajectory type: {type}")



    os.makedirs(out_folder, exist_ok=True)
    # make an empty engine to use for conversion
    setup = tps.TPS_setup(config)

    final_traj_storage_path = out_folder
    os.makedirs(final_traj_storage_path, exist_ok=True)
    paths.PathMover.engine = setup.md_engine
    logger.debug(f"Loading storage from {storage_path}...")
    storage = Storage(storage_path, "r")
    storage.engine = None
    logger.debug(f"Saving trajectories to {out_folder}...")
    save_rejected = setup.config.get("save_rejected", False)

    # reomve engine from storage 
    # storage.engine = None
    # steps = storage.steps
    # trajs = storage.trajectories
    logger.debug(f"Total number of steps in storage: {len(storage.steps)}")
    logger.debug(f"Total number of trajectories in storage: {len(storage.trajectories)}")
    end = len(storage.steps)
    logger.debug("Saving trajectories...")
    i = 0
    accepted = 0
    rejected = 0
    max_length_reject = 0
    shooting_traj = []
    atoms = read(setup.config.get("top_file","top.pdb"))
    traj_list = os.listdir(final_traj_storage_path)
    # traj_list = [f for f in traj_list if not f.startswith("._") and not f.endswith("accepted_0.xyz")]
    # import re

    # key = lambda s: int(re.search(r'\d+', s).group())    

    # traj_list = sorted(traj_list, key=key)
    for i in range(1000000):
        if f"accepted_{i}.xyz" in traj_list:
            accepted += 1
            continue
        elif f"rejected_max_{i}.xyz" in traj_list: 
            max_length_reject += 1
            continue
        elif f"rejected_same_{i}.xyz" in traj_list:
            rejected += 1
            continue #
        else:
            break
    logger.info(f"Resuming from index {i}...")
    start = i
    for idx  in range(start,end):
        try:
            shooting_snap = storage.steps[idx].change.canonical.details.shooting_snapshot
            top = shooting_snap.topology.mdtraj
            symbols = [atom.element.symbol if atom.element else atom.name[0] for atom in top.atoms]
            pos = shooting_snap.xyz*10  # nm to angstrom
            atoms = Atoms(positions=pos, symbols=symbols)
            
            if storage.steps[idx].change.accepted:
                atoms.info['shooting_state'] = 'accepted'
            else:
                if len(storage.trajectories[idx]) >= setup.n_frames_max:
                    atoms.info['shooting_state'] = 'rejected_max_length'
                else:
                    atoms.info['shooting_state'] = 'rejected_same_state'
                    
            shooting_traj.append(atoms)
        except Exception as e:
            logger.error(f"Error processing shooting snapshot for step {idx}: {e}")


        if Path(f"{final_traj_storage_path}/accepted_{i}.xyz").exists() or Path(f"{final_traj_storage_path}/rejected_max_{i}.xyz").exists() or Path(f"{final_traj_storage_path}/rejected_same_{i}.xyz").exists():
            logger.info(f"  Trajectory for step {i} already exists, skipping...")
            i += 1
            continue
            # 
            # break
        if storage.steps[idx].change.accepted:
            logger.debug(f"  Accepted trajectory for step {i}...")
            save_xyz(storage.trajectories[idx], f"{final_traj_storage_path}/accepted_{i}.xyz")
            logger.debug(f"Saved accepted trajectory for step {i}.")
            accepted += 1
        else:
            if len(storage.trajectories[idx]) >= setup.n_frames_max:
                logger.info(f" Rejected trajectory for step {i} -- max length reached.")
                if save_rejected:
                    save_xyz(storage.trajectories[idx], f"{final_traj_storage_path}/rejected_max_{i}.xyz",type="ase",topology=top)
                    logger.debug(f"Saved trajectory for step {i}.")     
                max_length_reject += 1
            else:
                logger.debug(f"  Rejected trajectory for step {i} -- same state ...")
                if save_rejected:
                    save_xyz(storage.trajectories[idx], f"{final_traj_storage_path}/rejected_same_{i}.xyz",type="ase",topology=top)
                    logger.debug(f"Saved trajectory for step {i}.")     
                rejected += 1

        i += 1
    write(f"{final_traj_storage_path}/shooting_traj.xyz", shooting_traj)
    logger.info(f"accepted: {accepted}, rejected: {rejected}, max length: {max_length_reject}")
    logger.info(f"acceptance rate: {np.round((end - (rejected+max_length_reject))/end, 2)}")
    storage.close()
    logger.debug("All trajectories saved.")
        

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--storage", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)

    args = p.parse_args()

    save_from_storage(
        args.storage,
        args.config,
        args.out,
    )



from openmm.app import PDBFile
from openmm import unit

def save_xyz_from_pdb(snapshot, pdb_file, fname):
    pdb = PDBFile(pdb_file)
    atoms = list(pdb.topology.atoms())
    pos = snapshot.xyz*10
    with open(fname, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom, p in zip(atoms, pos):
            elem = atom.element.symbol if atom.element else atom.name[0]
            f.write(f"{elem} {p[0]} {p[1]} {p[2]}\n")