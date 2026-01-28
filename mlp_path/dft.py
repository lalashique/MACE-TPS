from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Literal
import os 
from tqdm import tqdm
from loguru import logger

logger.add("dft_debug.log", level="INFO")

class DFTBase(ABC):
    """
    Abstract base class for all DFT engines.
    Provides unified interface, state handling, and capability validation.
    """
    supports_reftraj: bool = False  # overridden by subclasses

    def __init__(
        self,
        engine: str,
        settings: dict,
        input_files: list[str],
        reftraj: bool = False,
        run_mode: Literal["prepare", "run"] = "prepare",
        workdir: str = "output"
    ):
        self.engine = engine
        self.settings = settings
        self.reftraj = reftraj
        self.run_mode = run_mode
        self.workdir = Path(workdir)
        self.input_files = [Path(".")/f for f in input_files]
        self.completed = False
        self.results = None

        #initialise state
        self.state  = {
            "engine": self.engine,
            "run_mode": self.run_mode,
        }

        # ✅ validate capabilities on init
        self._validate_capabilities()

    def _validate_capabilities(self):
        if self.reftraj and not self.supports_reftraj:
            raise ValueError(
                f"[{self.engine.upper()}] reftraj mode is not supported. "
                f"Set reftraj=False or choose CP2K."
            )

    # ---------- public workflow ----------
    def run(self, structures):
        # for struct in structures:
        #     struct_name = os.path.basename(struct)
        #     self.state[struct_name] =  {"status": 0}  # 0: prepared, 1: run complete

        logger.debug(f"[BASE] Current state: {self.state}")

        self._prepare_structure(structures)
        if self.run_mode == "prepare":
            logger.debug(f"[BASE] {self.engine}: inputs prepared (mode={self._mode_name()}).")
            self._save_state()
            return

        self._run_calculation(structures)
        self._parse_outputs()
        self.completed = True
        self._save_state()
        print(f"[INFO] {self.engine}: run complete (mode={self._mode_name()}).")

    @abstractmethod
    def _prepare_structure(self, structures): ...
    @abstractmethod
    def _run_calculation(self, structures): ...
    @abstractmethod
    def _parse_outputs(self): ...

    def _mode_name(self):
        return "reftraj" if self.reftraj else "single"

    def _save_state(self):
        
        with open(self.workdir / "state.json", "w") as f:
            json.dump(self.state, f, indent=2)
    def _load_state(self):
        state_file = self.workdir / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")
        with open(state_file, "r") as f:
            state = json.load(f)
        return state

class CP2KDFT(DFTBase):
    supports_reftraj = True  # ✅ explicitly declare capability

    def __init__(self, settings, input_files, reftraj=False, run_mode="run", workdir="dft_cp2k"):
        super().__init__("cp2k", settings, input_files, reftraj, run_mode, workdir)

    def _prepare_structure(self, structures):
        input_dir = self.workdir
        input_dir.mkdir(parents=True, exist_ok=True)
        if self.reftraj:
            self._write_reftraj_input(input_dir, structures)
        else:
            self._write_single_inputs(input_dir, structures)
    def _write_reftraj_input(self, input_dir, structures):
        for i, struct in enumerate(structures):
            struct = Path(struct).resolve()
            input_dir_i = input_dir / f"structure_{i}"
            input_dir_i.mkdir(parents=True, exist_ok=True)
            # copy input files and coord into the made folder
            for file in self.input_files:
                dest_file = input_dir_i / file.name
                with open(file, "r") as src, open(dest_file, "w") as dst:
                    dst.write(src.read())
            coord_file = input_dir_i / "coord.xyz"
            # copy struct to coord.xyz using os module
            dest_file = coord_file
            dest_file.write_bytes(struct.read_bytes())
            # with open(coord_file, "w") as f:
            #     f.write(struct)  # assuming struct is a string of coordinates
        
            #make a blank file named PAUSE in the folder
            pause_file = input_dir_i / "PAUSE"
            with open(pause_file, "w") as f:
                f.write("")  # empty PAUSE file

            
            logger.debug(f"[CP2K] Preparing structure: {struct}")
            self.state[str(os.path.basename(input_dir_i))] =  {"status": 0, "file": str(struct)}  # 0: prepared, 1: run complete

            # Prepare CP2K reftraj input file
            logger.debug(f"[CP2K] Prepared reftraj input for structure {i} in {input_dir_i}")
            logger.info(f"Run using command: \n cd {input_dir_i} &&  {self.settings['command']} -i {self.input_files[0]} -o cp2k.out")
            # with open(input_file, "w") as f:
            #     f.write(f"! CP2K reftraj input for structure {i}\n")
                # ... rest of CP2K reftraj input writing logic
    def _run_calculation(self, structures):
        logger.debug(f"[CP2K] Running DFT calculation in {self.workdir}...")
        # Placeholder for actual CP2K execution logic using ase 
        from ase.calculators.cp2k import CP2K
        # for struct in structures:
        #     struct.set_calculator(CP2K(...))
        #     struct.get_potential_energy()
    def _parse_outputs(self):
        import MDAnalysis as mda
        import numpy as np
        from ase.io import read, write
        import cpoker.iparser as ip
        import cpoker.genfun as gf
        from ase import Atoms

        logger.debug(f"[CP2K] Parsing outputs in {self.workdir}...")
        for struct_key in self.state:
            if "structure_" not in struct_key:
                continue
            logger.debug(f"[CP2K] Parsing output for {Path(struct_key).resolve()}...")
            folder = str(Path(struct_key).resolve())
            u_pos = mda.Universe(f"{folder}/cp2k-pos-1.xyz")
            # Load forces (stored as XYZ coordinates)
            u_force = mda.Universe(f"{folder}/cp2k-frc-1.xyz")

            try:
                energies = np.loadtxt(f"{folder}/cp2k-1.ener")[:, 4]  # adjust if needed
            except Exception as e:
                logger.warning(f"Error loading energies: {e}, Taking energy from xyz file.")
                energies = np.array(gf.run_command(f"grep i {folder}/cp2k-pos-1.xyz | awk '{{print $NF}}'",out=True).split('\n')[:-1],dtype=float)
                logger.info(f"Number of energies loaded: {len(energies)}")
            
            # print(f"Number of energies loaded: {len(energies)}")

            inp = ip.iparser(folder).val_dict['ABC']
            cell = [float(x) for x in inp.split(" ")]

            logger.info(f"{len(u_pos.trajectory)} pos with {len(u_force.trajectory)} forces and {len(energies)} energies.")

            assert len(u_pos.trajectory) == len(u_force.trajectory) == len(energies)
            converge = gf.run_command(f"grep converge {folder}/cp2k.out ",out=True).split('\n')[:-1]
            # find how mnay converges mby looking for NOT in the lines, returning a true or false list
            converge_flags = np.array([ 'NOT' not in line for line in converge ])[:len(energies)]
            # log number of converged steps
            logger.info(f"Number of converged steps: {np.sum(converge_flags)} out of {len(converge_flags)}")
            


            atoms_list = []

            for ts_pos, ts_force, energy in tqdm(zip(u_pos.trajectory[converge_flags], u_force.trajectory[converge_flags], energies[converge_flags]),total=len(u_pos.trajectory[converge_flags])):
                symbols = u_pos.atoms.elements
                positions = ts_pos.positions
                forces = ts_force.positions*51.422   # forces stored as coords and converted to eV/Angstrom
                # Convert exact zeros to NaN
                forces[np.isclose(forces, 0.0)] = np.nan
                atoms = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=cell,
                    pbc=True
                )
                atoms.info['energy_cp2k'] = float(energy)*27.211  # convert to eV
                atoms.arrays['forces_cp2k'] = forces
                atoms_list.append(atoms)

            # save the file
            logger.info(f"Writing extended xyz to {os.path.join(folder, 'extdata.xyz')}")
            write(os.path.join(folder, 'extdata.xyz'), atoms_list, format='extxyz')
        # Placeholder for actual output parsing logic
        # self.results = {"energies": []}  # Dummy result

class DFTFactory:
    registry = {}

    @classmethod
    def register(cls, engine_name, cls_ref):
        cls.registry[engine_name.lower()] = cls_ref

    @classmethod
    def create(cls, engine, **kwargs):
        engine = engine.lower()
        if engine not in cls.registry:
            raise ValueError(f"Unknown DFT engine '{engine}'. Available: {list(cls.registry.keys())}")
        return cls.registry[engine](**kwargs)

# Register engines
DFTFactory.register("cp2k", CP2KDFT)
DFTFactory.register("vasp", None)  # Placeholder for VASPDFT class

if __name__ == "__main__":
    # Example usage
    logger.debug("Creating DFT engine...")
    dft_engine = DFTFactory.create(
        engine="cp2k",
        settings={"command": "mpirun -np 20 /home/ashiquel/software/cp2k-2024/cp2k_zen/cp2k-2024.1/exe/local/cp2k.popt"},
        input_files = ["cp2k.inp","force_eval.inp"],
        # reftraj=True,
        # run_mode="prepare",
        workdir="single_reftraj"
    )
    structures = ["coord.xyz","coord1.xyz"]  # Placeholder for actual structures
    logger.debug("DFT Engine Running...")

    dft_engine.run(structures)