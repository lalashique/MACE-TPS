# from mlp import create_mlp
# from tps import TPS_setup, Sampler

from loguru import logger
import json
from pathlib import Path

import os
import subprocess as sp
import shutil

from util.traj_util import select_frames

from ase.io import read, write
import numpy as np


def run_command(command, out=False):
    if out == True:
        try:
            output = sp.check_output(command, shell=True)
            output = output.decode("utf8").strip("/n")
        except sp.CalledProcessError:
            output = " "
        return output
    else:
        return sp.call(command, shell=True)

if __name__ == "__main__":
    logger.add("workflow_debug.log", level="DEBUG")

    dft_json_loc = "dft.json"  # Placeholder for actual DFT config file

    


    dft_json_loc = Path(dft_json_loc).resolve()
    logger.debug(f"[DFT] Reading DFT json file from {dft_json_loc}")
    
    with open(dft_json_loc, "r") as f:
        try:
            dft_config = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {dft_json_loc}: {e}")
            raise
    engine = dft_config.get("engine")
    settings = dft_config.get("settings")
    input_files = [Path(file).resolve() for file in  dft_config.get("input_files", [])]
    run_mode = dft_config.get("run_mode", "run")

    init_dir = Path(os.getcwd()).resolve()
    logger.debug(f"Initial working directory: {init_dir}")
    gen = 0
    while True:
        # check if done file exists in init_dir/gen
        done_file_gen = Path(init_dir) / str(gen) / "DONE"
        if not done_file_gen.exists():
            #  make folder name gen 
            os.makedirs(str(gen), exist_ok=True)
            logger.debug(f"Created directory for generation {gen}")
            os.chdir(str(gen))
            logger.debug(f"Changed working directory to: {os.getcwd()}")

            logger.debug("[DFT WORKFLOW] Starting DFT workflow...")

            os.makedirs("dft", exist_ok=True)
            os.chdir("dft")
            logger.debug(f"[DFT] Changed working directory to: {os.getcwd()}")
            # check for DONE file in dft folder
            done_file = Path("DONE").resolve()
            logger.debug(f"[DFT WORKFLOW] Checking for DONE file at {done_file}")
            dft_workdir = dft_config.get("workdir",os.getcwd())
            if not done_file.exists():

                logger.debug(f"[DFT] DFT Config: {dft_config}")
                logger.debug("[DFT] Creating DFT engine...")


                from dft import DFTFactory

                dft_engine = DFTFactory.create(
                        engine=engine,
                        settings=settings,
                        input_files = input_files,
                        reftraj=True,
                        run_mode=run_mode,
                        workdir=dft_workdir
                    )
                logger.debug("[DFT] DFT engine created.")
                if gen==0:
                    structures = dft_config.get("structures")  # Placeholder for actual structures
                    logger.debug(f"[DFT] DFT Engine Running on {structures}...")
                    structures = [Path(init_dir+ "/" + struct).resolve() for struct in structures]
                else:
                    structures = init_dir/"init"/"coord.xyz"
                    logger.debug(f"[DFT] DFT Engine Running on {structures}...")
                    #split coord.xyz to multiple for parallel dft calculations after asking how many splits
                    num_splits = int(input("Enter the number of splits for parallel DFT calculations: "))
                    # read coord.xyz using ase
                    all_frames = read(structures, index=":")
                    total_frames = len(all_frames)  
                    frames_per_split = total_frames // num_splits
                    # save splits to init_dir as coord{i}.xyz
                    for i in range(num_splits):
                        write(init_dir/"init"/f"coord{i}.xyz", all_frames[i*frames_per_split:(i+1)*frames_per_split])
                    structures = [init_dir/"init"/f"coord{i}.xyz" for i in range(num_splits)]
                logger.debug(f"[DFT] Structures for DFT: {structures}")

                # look for dft.json in the current directory
                dft_json_current = Path("state.json").resolve()
                logger.debug(f"[DFT] Looking for dft.json in current directory: {dft_json_current}")
                if not dft_json_current.exists():
                    dft_engine.run(structures)
                    for key in dft_engine.state:
                        if "structure_" in key:
                            logger.debug(f"Structure: {key}, State: {dft_engine.state[key]}")
                    logger.debug("[DFT] DFT set up complete.")
                    logger.debug("Please run the DFT calculations in the respective structure folders and then rerun this workflow script to continue.")
                    exit(0)
                else:
                    logger.debug(f"[DFT] dft.json already exists in current directory. Skipping DFT setup.")
                    logger.debug("[DFT] Loading previous DFT state. and checking if calculations are complete.")
                    dft_engine.state = dft_engine._load_state()
                for key in dft_engine.state:
                    logger.debug(f"Structure: {key}, State: {dft_engine.state[key]}")
                    if "structure_" in key:
                        # check if PAUSE file exists in the folder
                        struct_folder = Path(dft_engine.workdir) / key
                        pause_file = struct_folder / "PAUSE"
                        logger.debug(f"Checking PAUSE file for {key} at {pause_file}")
                        logger.debug(f"PAUSE file exists for {key}.")
                        #check if cp2k-pos-1.out contains same amount of lines as coord.xyz usinge grep and wc -l
                        coord_file = struct_folder / "coord.xyz"
                        out_file = struct_folder / "cp2k-pos-1.xyz"

                        if out_file.exists():
                            out_num = run_command(f"wc -l {struct_folder / 'cp2k-pos-1.xyz'}", out=True).split(" ")[0]
                            coord_num = run_command(f"wc -l {struct_folder / 'coord.xyz'}", out=True).split(" ")[0]
                            logger.debug(f"Lines in output: {out_num}, Lines in coord: {coord_num}")

                            # ask if want to remove PAUSE file if it exist
                            if pause_file.exists() and out_num == coord_num:
                                user_input = input(f"DFT calculation complete for {key}. Remove PAUSE file for {key}? (y/n): ")
                                if user_input.lower() == 'y':
                                    pause_file.unlink()
                            elif out_num == coord_num:
                                logger.debug(f"DFT calculation complete for {key}")
                            else:
                                logger.debug(f"DFT calculation not complete for {key}.")
                        else:
                            logger.debug(f"Output file {out_file} does not exist yet. Please run the DFT calculation.")
                            raise FileNotFoundError(f"Output file {out_file} not found.")
            
                dft_engine._parse_outputs()
                logger.debug("[DFT] DFT output parsing complete.")
                logger.debug("[DFT WORKFLOW] Combining the extxyz files into main directory...")

                # logger.debug("[DFT Workflow] Copying DFT outputs to main directory...")
                # copy all extxyz files from dft/workdir to init_dir/mlp/workdir

                dft_workdir_path = Path(dft_engine.workdir).resolve()
                dest_file = dft_workdir_path / "output" / "combined_extdata.xyz"
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Option 1: Using shutil.copyfileobj (fastest for multiple files)
                with open(dest_file, "wb") as dest_f:
                    for extxyz_file in dft_workdir_path.glob("structure_*/extdata.xyz"):
                        with open(extxyz_file, "rb") as src_f:
                            shutil.copyfileobj(src_f, dest_f)

                logger.debug(f"[MLP WORKFLOW] Combined extxyz files copied to {dest_file}")
                # touch a DONE file in dft_workdir to indicate combining is done  
                done_file = Path(dft_engine.workdir) / "DONE"
                done_file.touch()
                logger.debug(f"[DFT WORKFLOW] Created DONE file at {done_file}")
            else:
                logger.debug(f"[DFT WORKFLOW] DONE file exists in dft folder. Skipping DFT workflow.")
                
            logger.debug("Returning to initial directory...")
            os.chdir(init_dir)
            logger.debug(f"Current working directory: {os.getcwd()}")
            logger.debug("DFT workflow complete.")


            logger.debug("Starting MLP workflow...")
            mlp_workdir = Path(init_dir) / str(gen) / "mlp"
            mlp_workdir.mkdir(parents=True, exist_ok=True)

            logger.debug("[MLP] Starting MLP Setup.")
            os.chdir(mlp_workdir)

            #check for DONE file in mlp folder
            done_file = Path("DONE").resolve()
            logger.debug(f"[MLP WORKFLOW] Checking for DONE file at {done_file}")
            if not done_file.exists():
                logger.debug("[MLP Workflow] Copying DFT outputs to MLP directory...")
                # copy extxyz file from dft/output to mlp/data
                mlp_workdir = Path(init_dir) / str(gen) / "mlp"
                mlp_workdir.mkdir(parents=True, exist_ok=True)

                dest_file = mlp_workdir / "data" / "combined_extdata.xyz"
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                dft_workdir_path = Path(dft_workdir).resolve()
                dft_output_path = dft_workdir_path / "output" / "combined_extdata.xyz"
                shutil.copyfile(dft_output_path, dest_file)


                logger.debug(f"[MLP WORKFLOW] Combined extxyz files copied to {dest_file}")
                import util.traj_util as tu
                logger.debug(f"[MLP WORKFLOW] Splitting data into train and test sets...")
                tu.train_test_split(str(dest_file), test_fraction=0.25)
                logger.debug(f"[MLP WORKFLOW] Data split into train and test sets. ")

                logger.debug(f"[MLP] Changed working directory to: {os.getcwd()}")
                # TODO : setup MLP json in the mlp_workdir to check if it was previously set up and then skip setup if so
                # check if mlp.json exists in mlp_workdir

                #read mlp.json file from init_dir
                mlp_json_loc = Path(init_dir) / "mlp.json"
                logger.debug(f"[MLP] Reading MLP json file from {mlp_json_loc}")
                with open(mlp_json_loc, "r") as f:
                    try:
                        mlp_config = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from {mlp_json_loc}: {e}")
                        raise
                try:
                    E0s = mlp_config["E0s"]
                except KeyError:
                    logger.error("E0s not found in mlp.json")
                    raise KeyError("E0s not found in mlp.json")

                logger.debug(f"[MLP] MLP Config: {mlp_config}")

                import yaml
                # convert mlp_config to yaml and save as train.yaml in mlp_workdir
                train_yaml_loc = mlp_workdir / "train.yaml"
                with open(train_yaml_loc, "w") as f:
                    yaml.safe_dump(mlp_config, f)
                logger.debug(f"[MLP] MLP train.yaml saved to {train_yaml_loc}")

                logger.debug("[MLP] Creating MLP model...")    
                from mlp import create_mlp
                mlp = create_mlp("mace", "train.yaml", E0s=mlp_config["E0s"])
                mlp.train(train_file=str(dest_file.parent / "train.xyz"), test_file=str(dest_file.parent / "test.xyz"), seeds=[123, 234, 345], prepare_only=True,absolute_paths=True)
                logger.debug("[MLP] MLP model created and prepared for training.")
                
                # look for PAUSE files in mlp_workdir and remove them
                for pause_file in mlp_workdir.glob("runs/committee_*/PAUSE"):
                    logger.debug(f"PAUSE file: {pause_file}")
                    if pause_file.exists():
                        # look for MACE_models/mace01.model in the same committee folder
                        committee_folder = pause_file.parent
                        model_file = committee_folder / "MACE_models" / "mace01.model"
                        if model_file.exists():
                            logger.debug(f"Final Model file found: {model_file}")
                            user_input = input(f"Remove PAUSE file at {pause_file}? (y/n): ")
                            # if user_input.lower() == 'y':
                            #     pause_file.unlink()
                            #     logger.debug(f"Removed PAUSE file at {pause_file}")
                        else:
                            logger.debug(f"Final Model file not found: {model_file}. Keeping PAUSE file. Please complete training.")
                            raise 
                    # touch a DONE file in mlp_workdir to indicate mlp setup is done
                done_file = Path(mlp_workdir) / "DONE"
                done_file.touch()
            else:
                logger.debug(f"[MLP WORKFLOW] DONE file exists in mlp folder. Skipping MLP workflow.")
            logger.debug("MLP workflow complete.")

            logger.debug("[TPS WORKFLOW] Starting TPS workflow...")
            os.chdir(init_dir)
            logger.debug(f"[TPS WORKFLOW] Changed working directory to: {os.getcwd()}")  
            tps_workdir = Path(init_dir) / str(gen) / "tps"
            tps_workdir.mkdir(parents=True, exist_ok=True)
            os.chdir(tps_workdir)
            logger.debug(f"[TPS WORKFLOW] Changed working directory to: {os.getcwd()}")
            # check for DONE file in tps folder
            done_file = Path("DONE").resolve()
            logger.debug(f"[TPS WORKFLOW] Checking for DONE file at {done_file}")
            if not done_file.exists():
                logger.debug("Setting up TPS configuration...")
                #copy all the models to tps_workdir/models
                models_dir = tps_workdir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                for committee_folder in (mlp_workdir / "runs").glob("committee_*"):
                    model_file = committee_folder / "MACE_models" / "mace01.model"
                    dest_file = models_dir / f"{committee_folder.name}_mace01.model"
                    dest_file.write_bytes(model_file.read_bytes())
                    logger.debug(f"[TPS WORKFLOW] Copied model {model_file} to {dest_file}")
                #model path list
                model_paths = [str(models_dir / f"{committee_folder.name}_mace01.model") for committee_folder in (mlp_workdir / "runs").glob("committee_*")]
                logger.debug(f"[TPS WORKFLOW] Model paths for TPS: {model_paths}")
                logger.debug(f"[TPS WORKFLOW] All models copied to {models_dir}")
                logger.debug("[TPS WORKFLOW] Started Steered MD for initial path generation...")
                smd_workdir = tps_workdir / "smd"
                smd_workdir.mkdir(parents=True, exist_ok=True)
                os.chdir(smd_workdir)
                logger.debug(f"[SMD] Changed working directory to: {os.getcwd()}")
                
                #check if file exists smd.json in init_dir
                if not (Path(init_dir) / "smd.json").exists():
                    logger.error(f"[SMD] smd.json not found in {init_dir}. Cannot proceed with SMD setup.")
                    raise FileNotFoundError(f"smd.json not found in {init_dir}")
                source = Path(init_dir) / "smd.json"
                dest = Path('smd.json')
                dest.write_bytes(source.read_bytes())
                logger.debug(f"[SMD] Copied smd.json to {dest}")

                # check if smd.xyz exists in init_dir
                if not (Path(init_dir) / "smd.xyz").exists():
                    logger.error(f"[SMD] smd.xyz not found in {init_dir}. Cannot proceed with SMD setup.")
                    raise FileNotFoundError(f"smd.xyz not found in {init_dir}. Cannot proceed with SMD setup.")l
                source = Path(init_dir) / "smd.xyz"
                dest = Path('coord.xyz')
                dest.write_bytes(source.read_bytes())
                logger.debug(f"[SMD] Copied smd.xyz to {dest}")

                # check if plumed.dat exists in init_dir
                if not (Path(init_dir) / "plumed.dat").exists():
                    logger.error(f"[SMD] plumed.dat not found in {init_dir}. Cannot proceed with SMD setup.")
                    raise FileNotFoundError(f"plumed.dat not found in {init_dir}. Cannot proceed with SMD setup.")
                source = Path(init_dir) / "plumed.dat"
                dest = Path('plumed.dat')
                dest.write_bytes(source.read_bytes())
                logger.debug(f"[SMD] Copied plumed.dat to {dest}")

                # ask if run SMD now
                user_input = input(f"Run SMD now in {smd_workdir}? (y/N): ")
                if user_input.lower() == 'y':
                    logger.debug(f"[SMD] Exiting TPS workflow as per user request before SMD run.")
                    pause_file = smd_workdir / "PAUSE"
                    with open(pause_file, "w") as f:
                        f.write("")  # empty PAUSE file
                    logger.debug(f"[SMD] Created PAUSE file at {pause_file}")
                    logger.info(f"[SMD] Run SMD using command: \n cd {smd_workdir} && mlp run_md --model_path {model_paths[0]}  --md_params smd.json --plumed plumed.dat")

                    # check if PAUSE file exists in smd_workdir
                    if pause_file.exists():
                        logger.debug(f"[SMD] PAUSE file exists in {smd_workdir}. Please run the SMD simulation to proceed.")
                        exit(0)
                    else:
                        logger.debug(f"[SMD] PAUSE file not found in {smd_workdir}. SMD simulation may have already been run. ")
                        # check if tps.xyz exists in init_dir
                        if not (Path(init_dir) / "tps.xyz").exists():
                            logger.info(f"[SMD] Please verify and copy the resulting trajectory to the {init_dir} as tps.xyz to proceed with TPS sampling. ")
                            user_input = input(f"Continue? (y/N): ")
                            if user_input.lower() != 'y':
                                logger.debug(f"[SMD] Exiting TPS workflow as per user request.")
                                exit(0)
                        else:
                            logger.debug(f"[SMD] tps.xyz found in {init_dir}. Proceeding with TPS setup.")
                else:
                    #check if tps.xyz exists in init_dir
                    if not (Path(init_dir) / "tps.xyz").exists():
                        logger.error(f"[TPS] tps.xyz not found in {init_dir}. Cannot proceed with TPS setup.")
                        raise FileNotFoundError(f"tps.xyz not found in {init_dir}. Cannot proceed with TPS setup.")
                    else:
                        logger.debug(f"[TPS] tps.xyz found in {init_dir}. Proceeding with TPS setup.")
                os.chdir(tps_workdir)
                logger.debug(f"[TPS WORKFLOW] Changed working directory to: {os.getcwd()}")
                


                #check if file exists smd.json in init_dir
                if not (Path(init_dir) / "tps.json").exists():
                    logger.error(f"[TPS] tps.json not found in {init_dir}. Cannot proceed with TPS setup.")
                    raise FileNotFoundError(f"tps.json not found in {init_dir}")
                source = Path(init_dir) / "tps.json"
                dest = Path('tps.json')
                dest.write_bytes(source.read_bytes())
                logger.debug(f"[TPS] Copied tps.json to {dest}")

                #edit the tps.json to have model paths
                with open(dest, "r") as f:
                    try:
                        tps_config = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from {dest}: {e}")
                        raise
                tps_config["model_paths"] = model_paths[0]
                with open(dest, "w") as f:
                    json.dump(tps_config, f, indent=4) 
                logger.debug(f"[TPS] Updated tps.json with model paths.")

                    #check if file exists smd.json in init_dir
                if not (Path(init_dir) / "tps.xyz").exists():
                    logger.error(f"[TPS] tps.xyz not found in {init_dir}. Cannot proceed with TPS setup.")
                    raise FileNotFoundError(f"tps.xyz not found in {init_dir}. Cannot proceed with TPS setup.")
                source = Path(init_dir) / "tps.xyz"
                dest = Path('tps.xyz')
                dest.write_bytes(source.read_bytes())
                logger.debug(f"[TPS] Copied tps.xyz to {dest}")
                # from tps import TPS_setup, Sampler
                logger.debug("Setting up TPS sampler...")
                logger.info(f"""[TPS] To run TPS sampler, execute the following command:
                cd {tps_workdir} && tps_sampler""")

                logger.info (f"[TPS] TPS workflow setup complete. Please run the TPS sampler as indicated above to proceed with TPS sampling.")
                logger.info (f"[TPS] After running TPS sampler, please return here to continue the workflow.")
                # ask and continue to next steps
                user_input = input(f"Continue from TPS sampler now? (y/N): ")
                if user_input.lower() != 'y':
                    logger.debug(f"[TPS] Exiting TPS workflow as per user request before TPS run.")
                    exit(0)
                
                #check folder extend in tps_workdir/traj 
                traj_folder = tps_workdir / "traj"
                if not traj_folder.exists():
                    logger.error(f"[TPS] Trajectory folder not found: {traj_folder}. Cannot proceed with TPS analysis.")
                    raise FileNotFoundError(f"Trajectory folder not found: {traj_folder}")
                
                if not (traj_folder/"extend").exists():
                    logger.error(f"[TPS] Extend folder not found in trajectory folder: {traj_folder}. Cannot proceed with TPS analysis.")
                    raise FileNotFoundError(f"Extend folder not found in trajectory folder: {traj_folder}")

                # combine all extend traj files into one file in tps_workdir/traj/extend/combined_extend.xyz
                extend_folder = traj_folder / "extend"
                combined_extend_file = tps_workdir/"output" / "combined_extend.xyz"

                combined_extend_file.parent.mkdir(parents=True, exist_ok=True)

                # Option 1: Using shutil.copyfileobj (fastest for multiple files)
                with open(combined_extend_file, "wb") as dest_f:
                    for extxyz_file in extend_folder.glob("*_extended.xyz"):
                        with open(extxyz_file, "rb") as src_f:
                            shutil.copyfileobj(src_f, dest_f)

                logger.debug(f"[TPS WORKFLOW] Combined extend traj files into {combined_extend_file}")

                # select frames from combined_extend_file using util.traj_util.select_frames_onefile
                logger.debug(f"[TPS WORKFLOW] Selecting frames from combined extend traj...")
                out_file = tps_workdir / "output" / "selected_frames.xyz"
                select_frames(
                    xyz_path=str(combined_extend_file),
                    tps_params=tps_config,
                    out_file=str(out_file),
                )
                logger.debug(f"[TPS WORKFLOW] TPS frame selection complete.")

                # copy outfile to init_dir for next gen

                final_outfile = Path(init_dir) /str(gen)/ f"gen_{gen}_selected_frames.xyz"
                shutil.copyfile(out_file, final_outfile)
                logger.debug(f"[TPS WORKFLOW] Copied selected frames to {final_outfile} for next generation DFT calculations.")
                #touch a DONE file in tps_workdir to indicate tps analysis is done
                done_file = Path(tps_workdir) / "DONE"
                done_file.touch()
                logger.debug(f"[TPS WORKFLOW] Created DONE file at {done_file}")
            else:
                logger.debug(f"[TPS WORKFLOW] DONE file exists in tps folder. Skipping TPS workflow.")
            logger.debug("[TPS WORKFLOW] TPS workflow complete.")

            #touch a DONE file in init_dir/gen to indicate workflow gen is done
            done_file = Path(init_dir) / str(gen) / "DONE"
            done_file.touch()
            logger.debug(f"Workflow gen{gen} complete.")
            logger.debug(f"Moved back to initial directory: {os.getcwd()}")
            # backup init_dir/init  to init_dir/gen/init and clean coords for next gen
            backup_init_dir = Path(init_dir) / str(gen) / "init"
            if backup_init_dir.exists():
                shutil.rmtree(backup_init_dir)
            shutil.copytree(Path(init_dir) / "init", backup_init_dir)
            logger.debug(f"Backed up init directory to {backup_init_dir}")

            # remove all .xyz files from init_dir/init for next gen
            init_coords_dir = Path(init_dir) / "init"
            for xyz_file in init_coords_dir.glob("*.xyz"):
                xyz_file.unlink()
            logger.debug(f"Cleaned up .xyz files from {init_coords_dir} for next generation.")

            # copy gen_selected_frames.xyz to init_dir/init as coords for next gen dft
            selected_frames_file = Path(init_dir) / str(gen) / f"gen_{gen}_selected_frames.xyz"
            dest_file = init_coords_dir / "coord.xyz"
            shutil.copyfile(selected_frames_file, dest_file)
            logger.debug(f"Copied selected frames to {dest_file} for next generation DFT calculations.")
            gen += 1
            # ask if continue to next gen
            user_input = input(f"Continue to next generation {gen}? (y/N): ")
        else:
            logger.debug(f"DONE file exists for generation {gen}. Skipping entire workflow for this generation.")
            gen += 1
