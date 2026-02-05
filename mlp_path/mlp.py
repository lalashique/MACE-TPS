import sys
import logging
import warnings
from abc import ABC, abstractmethod
from loguru import logger
from pathlib import Path
import yaml
import os
import time

import numpy as np

warnings.filterwarnings("ignore")

# --- Base class ---
class MLP(ABC):
    def __init__(self, config_file):
        self.config_file = Path(config_file) if config_file else None
        self.model = None

    @abstractmethod
    def train(self, seed=None, output_dir=None):
        pass

    @abstractmethod
    def load(self, model_path):
        pass

    @abstractmethod
    def predict(self, data):
        pass
    
    # @abstractmethod 
    # @staticmethod
    # def run_md(model_paths, structure, md_params):
    #     pass

# --- MACE implementation ---
class MACE_MLP(MLP):
    DEFAULT_CONFIG = {
        "model": "MACE",
        "num_channels": 32,
        # "max_L": 0,
        # "correlation": 3,
        "r_max": 4.5,
        "name": "mace01",
        "model_dir": "MACE_models",
        "log_dir": "MACE_models",
        "checkpoints_dir": "chkpt",
        "results_dir": "MACE_models",
        "train_file": "data/train.xyz",
        "valid_fraction": 0.10,
        "test_file": "data/test.xyz",
        "energy_key": "energy_cp2k",
        "forces_key": "forces_cp2k",
        "E0s": "{1: -13.50896562 , 6: -145.7376780, 8: -428.8933516,46: -3454.585309 , 78: -3262.604121 }",
        "device": "cuda",
        "batch_size": 4,
        "max_num_epochs": 100,
        "lr": 0.01,
        "weight_decay": 5e-5,
        "swa": True,
        "energy_weight": 100,
        "forces_weight": 1000,
        "seed": 123,
        "save_cpu": True,
        "error_table": "PerAtomRMSE",
    }

    def __init__(self, config_file=None,E0s=None):
        super().__init__(config_file)
        self.config = self._get_config()

        self.E0s = E0s
            
        # from mace.cli.run_train import main as mace_train
        # from mace.calculators import mace_calculator
        # self._train_func = mace_train
        # self._calc_cls = mace_calculator
        self.committee_models = []

    def _get_config(self):
        if self.config_file and self.config_file.exists():
            with open(self.config_file) as f:
                return yaml.safe_load(f)
        return dict(self.DEFAULT_CONFIG)

    def train(
        self,
        seeds=None,
        output_dir="runs",
        train_file=None,
        test_file=None,
        train_test_split=None,
        prepare_only=False,
        absolute_paths=False,
        hpc_command_template="python -m mace.cli.run_train --config {config}",
    ):  
        """Train one or multiple models (committee)."""
        if seeds is None:
            seeds = [self.DEFAULT_CONFIG["seed"]]
        if not isinstance(seeds, (list, tuple)):
            seeds = [seeds]

        if not test_file and train_test_split:
            raise NotImplementedError("train_test_split is not implemented yet.")

        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        if train_file:
            train_file = Path(train_file)
            test_file = Path(test_file) 
            if absolute_paths:
                train_file_path = os.path.abspath(train_file)
                test_file_path = os.path.abspath(test_file)
            else:
                raise NotImplementedError
        trained_models = []
        hpc_cmds = []

        for i, seed in enumerate(seeds):
            member_dir = base_dir / (f"committee_{i:02d}" if len(seeds) > 1 else "")
            cmd = self._prepare_member(seed, member_dir, train_file, test_file)
            hpc_cmds.append(cmd)

            if not prepare_only:
                self._run_training(cmd)
                model_path = next(member_dir.glob("*.model"), None)
                if model_path:
                    trained_models.append(self._calc_cls(model_path))

        # if preparing only, print out HPC commands
        if prepare_only:
            print("\nHPC training commands:")
            for cmd in hpc_cmds:
                print(cmd)
            return

        self.committee_models = trained_models
        if len(seeds) == 1 and trained_models:
            self.model = trained_models[0]
        return trained_models if len(seeds) > 1 else trained_models[0]

    def _prepare_member(self, seed, output_dir, train_file=None, test_file=None):

        cfg = self.config  # make a copy
        cfg["seed"] = seed

        if self.E0s:
            cfg["E0s"] = self.E0s
        else:
            raise ValueError("E0s must be provided for training.")
        

        if train_file:
            cfg["train_file"] = str(Path(train_file).absolute())
        if test_file:
            cfg["test_file"] = str(Path(test_file).absolute())

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        # cfg["model_dir"] = str(out)
        # cfg["log_dir"] = str(out)
        # cfg["results_dir"] = str(out)
        # cfg["checkpoints_dir"] = str(out / "chkpt")

        tmp_cfg = out / f"train_seed{seed}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

        # put a PAUSE file in the output dir to indicate preparation is done
        pause_file = out / "PAUSE"
        with open(pause_file, "w") as f:
            f.write("")  # empty PAUSE file
        
        out = out.resolve()

        return f"cd {out} && python -m mace.cli.run_train --config train_seed{seed}.yaml"

    def _run_training(self, command):
        logging.getLogger().handlers.clear()
        sys.argv = command.split()
        self._train_func()

    def load(self, model_path_or_dir):
        p = Path(model_path_or_dir)
        if p.is_dir():
            models = list(p.glob("*.model"))
            if not models:
                raise FileNotFoundError(f"No model found in {p}")
            p = models[0]
        self.model = self._calc_cls(p)
        return self.model

    def load_committee(self, base_dir):
        base = Path(base_dir)
        members = sorted(base.glob("committee_*"))
        self.committee_models = [
            self._calc_cls(next(m.glob("*.model"))) for m in members if list(m.glob("*.model"))
        ]
        return self.committee_models

    def predict(self, data, committee=False, return_std=False):
        if committee and self.committee_models:
            preds = np.array([m.calculate(data)["energy"] for m in self.committee_models])
            mean = preds.mean(axis=0)
            if return_std:
                std = preds.std(axis=0)
                return mean, std
            return mean
        elif self.model:
            return self.model.calculate(data)
        else:
            raise RuntimeError("No model loaded.")
    @staticmethod
    def run_md(model_path, md_params,plumed=None,type= "local", esen_foundational_model=False):
        from ase.io import read
        
        from ase import Atoms
        from ase.calculators.plumed import Plumed
        from ase import units
        import json 


        if type == "local":
            try:
                model_path = model_path if isinstance(model_path, Path) else Path(model_path)
            except Exception as e:
                raise ValueError(f"Invalid model path: {model_path}") from e

        elif type == "universal":
            raise NotImplementedError("Universal type MD is not implemented yet.")
        
        try:
            with open(md_params, 'r') as f:
                md_params = json.load(f)
        except Exception as e:
            raise ValueError(f"Error reading md_params: {md_params}") from e
        initial_time = time.time()
        restart = md_params.get("restart", False)
        initalize_velocities = md_params.get("initialize_velocities", True)
        if restart:
            if not Path('md_out.xyz').exists():
                raise FileNotFoundError("md_out.xyz not found for restart.")
            init_conf = read('md_out.xyz', -1)
            logger.debug(f"Restarting MD from md_out.xyz with {len(init_conf)} atoms.")
        else:   
            init_conf = read('coord.xyz', '0')
        if "cell" in md_params:
            logger.debug(f"Setting cell from md_params{np.diag(np.array(md_params['cell'],dtype=float))}.")
            init_conf.cell = np.diag(np.array(md_params["cell"],dtype=float))
            init_conf.center()
            init_conf.set_pbc((True, True, True))
        elif init_conf.cell is not None and not init_conf.cell.any() == 0:
            logger.debug(f"Using cell from coord file: {init_conf.cell}.")
            init_conf.set_pbc((True, True, True))
        else:
            raise ValueError("Cell parameters are missing in both the coord file and md_params.")
        # init_conf.cell = ([ 11.24800,9.7410,40])
        # init_conf.set_pbc((True, True, True))
        print(init_conf)
        #using one of the esen foundation models
        if esen_foundational_model:
            from fairchem.core import FAIRChemCalculator
            calculator = FAIRChemCalculator.from_model_checkpoint(name_or_path=str(model_path), device="cuda")
        else:
            from mace.calculators import MACECalculator
            calculator = MACECalculator(model_paths= model_path, device='cuda')
        # init_conf.set_calculator(calculator)
        # check if timestep 
        timestep = float(md_params.get("time_step",1.0))*units.fs  # default 1 fs
        logger.debug(f"MD Timestep: {float(md_params.get('time_step',1.0))} fs")
        ps = 1000 * units.fs
        logger.debug(f"MD Timestep: {ps} ps")
        if plumed:
            plumed = Path(plumed)
            #check if plumed file exists
            if not plumed.exists():
                raise FileNotFoundError(f"Plumed file not found: {plumed}")
            setup = [f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}"]
            if restart:
                setup.append("RESTART")
            setup.append(f'INCLUDE FILE={str(plumed)}')

            init_conf.calc = Plumed(calc=calculator,
                                    input=setup,
                                    timestep=timestep/units.fs/1000,
                                    atoms=init_conf,
                                    kT=0.03)
        else:
            init_conf.calc = calculator
        
        from ase.md import MDLogger
        from ase.md.langevin import Langevin
        temperature = md_params.get("temperature", 330)  # default 300 K
        friction = md_params.get("friction", 5e-3)  # default 5e-3 1/fs

        thermostat = md_params.get("thermostat", "Langevin")
        if thermostat == "Langevin":
            logger.debug(f"Using Langevin thermostat with friction {friction} 1/fs.")
            dyn = Langevin(init_conf, timestep, temperature_K=temperature, friction=friction,logfile='md_out.log')
        elif thermostat == "NoseHooverChain":
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
            dyn = NoseHooverChainNVT(init_conf, timestep, temperature_K=temperature, tdamp=50* units.fs, tchain=4,logfile='md_out.log')
        elif thermostat == "Berendsen":
            from ase.md.berendsen import Berendsen
            tau = md_params.get("tau", 25*timestep)  # default 25*timestep
            logger.debug(f"Using Berendsen thermostat with tau {tau} fs.")
            dyn = Berendsen(init_conf, timestep, temperature_K=temperature, taut= tau,logfile='md_out.log')
        elif thermostat == "VelocityVerlet":
            from ase.md.velocityverlet import VelocityVerlet
            logger.debug(f"Using Velocity Verlet integrator (NVE).")
            dyn = VelocityVerlet(init_conf, timestep,logfile='md_out.log')
        else:
            raise ValueError(f"Unknown thermostat: {thermostat}")
        
        def write_frame():
                dyn.atoms.write('md_out.xyz', append=True)
        def write_frame_small():
                dyn.atoms.write('md_out_small.xyz', append=True)
        step = 0
        def write_time():
                nonlocal step
                if step == 0:
                        step = time.time()
                time_per_step = time.time() - step
                with open('md_out.time', 'a') as f:
                        f.write(str(time_per_step)+'\n')
                step = time.time()

        interval = int(md_params.get("interval", 1))
        small_interval = int(md_params.get("small_interval", 50))

        if initalize_velocities and restart== False:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            MaxwellBoltzmannDistribution(init_conf, temperature_K=temperature)
            logger.debug(f"Initialized velocities at {temperature} K using Maxwell-Boltzmann distribution.")
        elif restart:
            logger.debug(f"Restarting MD from existing velocities.")
            init_conf.set_velocities(init_conf.get_velocities())
        else:
            logger.debug(f"Velocities not initialized.")

        dyn.attach(write_frame, interval=interval)
        dyn.attach(write_frame_small, interval=small_interval)
        dyn.attach(write_time, interval=1)

        total_steps = int(md_params.get("total_steps", 15000))
        dyn.run(total_steps)
        print("MD finished!")
        pause_file = Path("PAUSE")
        if pause_file.exists():
            pause_file.unlink()

        with open('md_out.time', 'a') as f:
            f.write(str(time.time() - initial_time)+'\n')
    
    @staticmethod
    def run_minimization(model_path, md_params,type= "local"):
        from ase.io import read
        from mace.calculators import MACECalculator
        from ase import Atoms
        from ase.optimize import BFGS
        import json 

        if type == "local":
            try:
                model_path = model_path if isinstance(model_path, Path) else Path(model_path)
            except Exception as e:
                raise ValueError(f"Invalid model path: {model_path}") from e

        elif type == "universal":
            raise NotImplementedError("Universal type minimization is not implemented yet.")
        try:
            with open(md_params, 'r') as f:
                md_params = json.load(f)
        except Exception as e:
            raise ValueError(f"Error reading md_params: {md_params}") from e
        
        init_conf = read('coord.xyz', '0')

        #check if cell is present in the xyz file
        if "cell" in md_params:
            logger.debug(f"Setting cell from md_params{np.diag(np.array(md_params['cell'],dtype=float))}.")
            init_conf.cell = np.diag(np.array(md_params["cell"],dtype=float))
            init_conf.center()
            init_conf.set_pbc((True, True, True))
        elif init_conf.cell is not None and not init_conf.cell.any() == 0:
            logger.debug(f"Using cell from coord file: {init_conf.cell}.")
            init_conf.set_pbc((True, True, True))
        else:
            raise ValueError("Cell parameters are missing in both the coord file and md_params.")

        calculator = MACECalculator(model_paths= model_path, device='cuda')
        init_conf.set_calculator(calculator)
        def write_frame():
                dyn.atoms.write('md_out.xyz', append=True)
        dyn = BFGS(init_conf, trajectory='minimization.traj')
        max_steps = int(md_params.get("max_steps", 1000))
        fmax = float(md_params.get("fmax", 0.01))
        interval = int(md_params.get("interval", 1))
        dyn.attach(write_frame, interval=interval)

        dyn.run(fmax=fmax, steps=max_steps)
        print("Minimization finished!")

    @staticmethod
    def evaluate_committee(atoms,committee_calcs):
        """Compute committee energy/force variances and store in atoms.info/arrays."""
        main_calc = atoms.calc  # backup
        energies, forces = [], []
        for i, calc in enumerate(committee_calcs):
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
            atoms.info[f"energy_m{i+1}"] = energies[-1]
            if i == 0:
                atoms.arrays["forces"] = forces[-1]
            # del atoms.arrays['momenta']

        energies = np.array(energies)
        forces = np.stack(forces)
        atoms.info["energy_variance"] = np.var(energies)
        atoms.info["force_variance"] = np.var(forces, axis=0).mean()
        atoms.arrays["force_var_per_atom"] = np.var(forces, axis=0).mean(axis=1)
        atoms.calc = main_calc  # restore
    @staticmethod
    def evaluate(model_path, input_file,output_file,type= "local"):
        from ase.io import read, write
        from mace.calculators import MACECalculator
        from ase import Atoms
        import copy

        if type == "local":
            try:
                model_path = model_path if isinstance(model_path, Path) else Path(model_path)
            except Exception as e:
                raise ValueError(f"Invalid model path: {model_path}") from e

        elif type == "universal":
            raise NotImplementedError("Universal type evaluation is not implemented yet.")
        
        structures = read(input_file, ':')
        calculator = MACECalculator(model_paths= model_path, device='cuda')
        new_structures = []
        results = []
        for i, atoms in enumerate(structures):
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            if len(atoms) != forces.shape[0]:
                raise ValueError(f"Number of atoms ({len(atoms)}) does not match number of forces ({forces.shape[0]}).")
            atoms.info['predicted_energy'] = energy
            atoms.arrays['predicted_forces'] = forces
            new_structures.append(atoms.copy())
            results.append((energy, forces))
            print(f"Structure {i}: Energy = {energy} eV")
            # break

        write(output_file, new_structures)
        print(f"Results written to {output_file}")
        return results

# --- Factory helper ---
def create_mlp(method,*args, **kwargs):
    if method.lower() == "mace":
        return MACE_MLP(*args, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    import argparse
    import inspect
    p = argparse.ArgumentParser()

    # pick which method
    p.add_argument("--method", choices=["run_md","run_minimization","evaluate"], nargs="?")
    args, remaining = p.parse_known_args()
    logger.debug(f"Arguments: {args}, Remaining: {remaining}")
    if args.method:
        # fallback / default behavior

        # otherwise run the selected static method
        fn = getattr(MACE_MLP, args.method, None)
        if fn:
            # get the function
            fn = getattr(MACE_MLP, args.method)

            # inspect args
            sig = inspect.signature(fn)

            # build dynamic parser for this method
            p2 = argparse.ArgumentParser()
            for name, param in sig.parameters.items():
                if param.default is inspect._empty:
                    p2.add_argument(f"--{name}", required=True)
                else:
                    p2.add_argument(f"--{name}", default=param.default)

            args2 = p2.parse_args(remaining)

            # call function
            kwargs = vars(args2)
            out = fn(**kwargs)
        else:
            print(f"unknown method: {args.method}")
            exit(1)
    else:
        print("no method given → running default stuff")
        # REMOVE BELOW: example usage
        # mlp = create_mlp("mace", "train.yaml",
        #                 E0s={1: -13.52638614, 6: -145.7828756, 8: -429.0792406, 29: -1303.851651})
        # mlp.train(train_file="train.xyz", test_file="test.xyz", seeds=[123, 234, 345], prepare_only=True,absolute_paths=True)

