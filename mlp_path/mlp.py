"""
mlp.py — Machine Learning Potential interface for the mlpath workflow.

Provides an abstract base (MLP) and a concrete MACE implementation (MACE_MLP).
Supports training committees of models (same architecture, different seeds) for
uncertainty quantification, plus MD, minimisation, and evaluation utilities.

Typical usage
-------------
    from mlp import create_mlp

    mlp = create_mlp("mace", "train.yaml", E0s={1: -13.5, 8: -428.9})
    mlp.train(
        train_file="data/train.xyz",
        test_file="data/test.xyz",
        seeds=[123, 234, 345],
        prepare_only=True,   # write configs + PAUSE; submit jobs externally
        absolute_paths=True,
    )

CLI usage
---------
    python mlp.py --method run_md --model_path model.model --md_params md.json
    python mlp.py --method evaluate --model_path model.model \
                  --input_file data.xyz --output_file out.xyz
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MLP(ABC):
    """Abstract base class for machine learning potential wrappers."""

    def __init__(self, config_file: str | Path | None) -> None:
        self.config_file = Path(config_file) if config_file else None
        self.model = None

    @abstractmethod
    def train(self, **kwargs): ...

    @abstractmethod
    def load(self, model_path: str | Path): ...

    @abstractmethod
    def predict(self, data): ...


# ---------------------------------------------------------------------------
# MACE implementation
# ---------------------------------------------------------------------------

class MACE_MLP(MLP):
    """
    Wrapper around the MACE interatomic potential.

    Supports:
    - Training one or more committee models (different random seeds).
    - Loading a single model or an entire committee from a directory.
    - Committee-based uncertainty estimation (energy/force variance).
    - Running NVT/NVE molecular dynamics via ASE.
    - Running BFGS geometry minimisation via ASE.
    - Evaluating a model on a structure file and writing predictions.

    Parameters
    ----------
    config_file : str or Path, optional
        Path to a YAML config file.  If not provided the built-in defaults
        are used.
    E0s : dict, optional
        Isolated-atom reference energies {atomic_number: energy_eV}.
        Required for training.
    """

    DEFAULT_CONFIG: dict = {
        "model": "MACE",
        "num_channels": 32,
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
        "E0s": "{}",
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

    def __init__(
        self,
        config_file: str | Path | None = None,
        E0s: dict | None = None,
    ) -> None:
        super().__init__(config_file)
        self.config = self._load_config()
        self.E0s = E0s
        self.committee_models: list = []

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        if self.config_file and self.config_file.exists():
            with open(self.config_file) as f:
                return yaml.safe_load(f)
        return dict(self.DEFAULT_CONFIG)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        seeds: int | list[int] | None = None,
        output_dir: str = "runs",
        train_file: str | None = None,
        test_file: str | None = None,
        prepare_only: bool = False,
        absolute_paths: bool = False,
        hpc_command_template: str = "python -m mace.cli.run_train --config {config}",
    ) -> list | object | None:
        """
        Prepare training configs and (optionally) launch training.

        Parameters
        ----------
        seeds : int or list[int], optional
            Random seed(s).  One committee member is trained per seed.
        output_dir : str
            Root directory for committee member subdirectories.
        train_file : str, optional
            Path to training XYZ file.
        test_file : str, optional
            Path to test XYZ file.
        prepare_only : bool
            If True, write configs and print HPC commands but do not run.
        absolute_paths : bool
            Store absolute paths in the generated YAML configs.
        hpc_command_template : str
            Template for the HPC submission command (unused if prepare_only=False).

        Returns
        -------
        Trained model(s) or None if prepare_only=True.
        """
        if seeds is None:
            seeds = [self.DEFAULT_CONFIG["seed"]]
        if isinstance(seeds, int):
            seeds = [seeds]

        if not self.E0s:
            raise ValueError("E0s must be provided for training.")
        if not absolute_paths and train_file:
            raise NotImplementedError(
                "Relative paths are not supported; set absolute_paths=True."
            )

        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        hpc_cmds: list[str] = []
        trained_models: list = []

        for i, seed in enumerate(seeds):
            member_dir = base_dir / (f"committee_{i:02d}" if len(seeds) > 1 else "")
            cmd = self._prepare_member(seed, member_dir, train_file, test_file)
            hpc_cmds.append(cmd)

            if not prepare_only:
                self._run_training(cmd)
                model_path = next(member_dir.glob("*.model"), None)
                if model_path:
                    from mace.calculators import MACECalculator
                    trained_models.append(MACECalculator(model_paths=model_path, device="cuda"))

        if prepare_only:
            logger.info("HPC training commands:")
            for cmd in hpc_cmds:
                logger.info(f"  {cmd}")
            return None

        self.committee_models = trained_models
        if len(seeds) == 1 and trained_models:
            self.model = trained_models[0]
        return trained_models if len(seeds) > 1 else (trained_models[0] if trained_models else None)

    def _prepare_member(
        self,
        seed: int,
        output_dir: Path,
        train_file: str | None,
        test_file: str | None,
    ) -> str:
        cfg = dict(self.config)
        cfg["seed"] = seed
        cfg["E0s"] = str(self.E0s)

        if train_file:
            cfg["train_file"] = str(Path(train_file).resolve())
        if test_file:
            cfg["test_file"] = str(Path(test_file).resolve())

        output_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = output_dir / f"train_seed{seed}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg, f)

        (output_dir / "PAUSE").touch()

        return f"cd {output_dir.resolve()} && python -m mace.cli.run_train --config {yaml_path.name}"

    def _run_training(self, command: str) -> None:
        from mace.cli.run_train import main as mace_train
        logging.getLogger().handlers.clear()
        sys.argv = command.split()
        mace_train()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_path_or_dir: str | Path) -> object:
        """Load a single MACE model (from a file or the first .model in a dir)."""
        from mace.calculators import MACECalculator
        p = Path(model_path_or_dir)
        if p.is_dir():
            candidates = list(p.glob("*.model"))
            if not candidates:
                raise FileNotFoundError(f"No .model file found in {p}")
            p = candidates[0]
        self.model = MACECalculator(model_paths=p, device="cuda")
        return self.model

    def load_committee(self, base_dir: str | Path) -> list:
        """Load all committee models from ``base_dir/committee_*/``."""
        from mace.calculators import MACECalculator
        base = Path(base_dir)
        self.committee_models = []
        for member in sorted(base.glob("committee_*")):
            model_file = next(member.glob("*.model"), None)
            if model_file:
                self.committee_models.append(
                    MACECalculator(model_paths=model_file, device="cuda")
                )
        return self.committee_models

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, data, committee: bool = False, return_std: bool = False):
        """
        Predict energy (and optionally std) for ``data``.

        Parameters
        ----------
        data : ASE Atoms
            Input structure.
        committee : bool
            Average over all committee models.
        return_std : bool
            Also return the standard deviation across committee predictions.
        """
        if committee and self.committee_models:
            preds = np.array([m.calculate(data)["energy"] for m in self.committee_models])
            mean = preds.mean(axis=0)
            return (mean, preds.std(axis=0)) if return_std else mean
        if self.model:
            return self.model.calculate(data)
        raise RuntimeError("No model loaded. Call load() or train() first.")

    # ------------------------------------------------------------------
    # Committee uncertainty
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_committee(atoms, committee_calcs: list) -> None:
        """
        Run all committee models on ``atoms`` and store variance statistics in-place.

        Adds to ``atoms.info``:
        - ``energy_m{i}``     — per-model energy
        - ``energy_variance`` — variance across models
        - ``force_variance``  — mean force variance across atoms and models

        Adds to ``atoms.arrays``:
        - ``forces``           — forces from model 0
        - ``force_var_per_atom`` — per-atom mean force variance
        """
        backup_calc = atoms.calc
        energies: list[float] = []
        forces: list[np.ndarray] = []

        for i, calc in enumerate(committee_calcs):
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            energies.append(e)
            forces.append(f)
            atoms.info[f"energy_m{i + 1}"] = e
            if i == 0:
                atoms.arrays["forces"] = f

        e_arr = np.array(energies)
        f_arr = np.stack(forces)          # shape (n_models, n_atoms, 3)

        atoms.info["energy_variance"]     = float(np.var(e_arr))
        atoms.info["force_variance"]      = float(np.var(f_arr, axis=0).mean())
        atoms.arrays["force_var_per_atom"] = np.var(f_arr, axis=0).mean(axis=1)

        atoms.calc = backup_calc

    # ------------------------------------------------------------------
    # Molecular dynamics
    # ------------------------------------------------------------------

    @staticmethod
    def run_md(
        model_path: str | Path,
        md_params: str | Path,
        plumed: str | Path | None = None,
        type: str = "local",
        esen_foundational_model: bool = False,
    ) -> None:
        """
        Run NVT (or NVE) molecular dynamics using ASE.

        Parameters
        ----------
        model_path : str or Path
            Path to the trained MACE model (or FAIRChem checkpoint if
            ``esen_foundational_model=True``).
        md_params : str or Path
            Path to a JSON file with MD settings.  Keys:

            - ``restart``            (bool, default False)
            - ``initialize_velocities`` (bool, default True)
            - ``cell``               (list[float], optional)
            - ``time_step``          (float, fs, default 1.0)
            - ``temperature``        (float, K, default 330)
            - ``friction``           (float, 1/fs, default 5e-3)
            - ``thermostat``         (str, default "Langevin")
            - ``total_steps``        (int, default 15000)
            - ``interval``           (int, default 1)
            - ``small_interval``     (int, default 50)

        plumed : str or Path, optional
            Path to a PLUMED input file for biased MD.
        type : {"local", "universal"}
            "local" uses the provided model; "universal" is not yet implemented.
        esen_foundational_model : bool
            Use a FAIRChem foundation model instead of MACE.
        """
        from ase import units
        from ase.io import read
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        if type != "local":
            raise NotImplementedError(f"MD type '{type}' is not yet implemented.")

        model_path = Path(model_path)
        with open(md_params) as f:
            params = json.load(f)

        initial_time = time.time()
        restart             = params.get("restart", False)
        init_velocities     = params.get("initialize_velocities", True)
        temperature         = params.get("temperature", 330)
        friction            = params.get("friction", 5e-3)
        timestep_fs         = float(params.get("time_step", 1.0))
        timestep            = timestep_fs * units.fs
        total_steps         = int(params.get("total_steps", 15000))
        interval            = int(params.get("interval", 1))
        small_interval      = int(params.get("small_interval", 50))
        thermostat_name     = params.get("thermostat", "Langevin")

        # Load initial configuration
        if restart:
            if not Path("md_out.xyz").exists():
                raise FileNotFoundError("md_out.xyz not found for restart.")
            init_conf = read("md_out.xyz", -1)
            logger.debug("Restarting MD from md_out.xyz.")
        else:
            init_conf = read("coord.xyz", 0)

        # Set periodic boundary / cell
        if "cell" in params:
            init_conf.cell = np.diag(np.array(params["cell"], dtype=float))
            init_conf.center()
            init_conf.set_pbc(True)
        elif init_conf.cell is not None and init_conf.cell.any():
            init_conf.set_pbc(True)
        else:
            raise ValueError("Cell parameters missing in both coord.xyz and md_params.")

        # Calculator
        if esen_foundational_model:
            from fairchem.core import FAIRChemCalculator
            calculator = FAIRChemCalculator.from_model_checkpoint(
                name_or_path=str(model_path), device="cuda"
            )
        else:
            from mace.calculators import MACECalculator
            calculator = MACECalculator(model_paths=model_path, device="cuda")

        # Optional PLUMED bias
        if plumed:
            from ase.calculators.plumed import Plumed
            plumed = Path(plumed)
            if not plumed.exists():
                raise FileNotFoundError(f"PLUMED file not found: {plumed}")
            ps = 1000 * units.fs
            setup = [f"UNITS LENGTH=A TIME={1 / ps} ENERGY={units.mol / units.kJ}"]
            if restart:
                setup.append("RESTART")
            setup.append(f"INCLUDE FILE={plumed}")
            init_conf.calc = Plumed(
                calc=calculator,
                input=setup,
                timestep=timestep_fs / 1000,
                atoms=init_conf,
                kT=0.03,
            )
        else:
            init_conf.calc = calculator

        # Thermostat / integrator
        dyn = MACE_MLP._build_thermostat(
            thermostat_name, init_conf, timestep, temperature, friction, params
        )

        # Velocity initialisation
        if init_velocities and not restart:
            MaxwellBoltzmannDistribution(init_conf, temperature_K=temperature)
            logger.debug(f"Velocities initialised at {temperature} K (Maxwell-Boltzmann).")
        elif restart:
            init_conf.set_velocities(init_conf.get_velocities())

        # Attach trajectory writers and timer
        step_timer = {"last": time.time()}

        def _write_frame():
            dyn.atoms.write("md_out.xyz", append=True)

        def _write_frame_small():
            dyn.atoms.write("md_out_small.xyz", append=True)

        def _write_time():
            now = time.time()
            with open("md_out.time", "a") as f:
                f.write(f"{now - step_timer['last']}\n")
            step_timer["last"] = now

        dyn.attach(_write_frame,       interval=interval)
        dyn.attach(_write_frame_small, interval=small_interval)
        dyn.attach(_write_time,        interval=1)

        dyn.run(total_steps)
        logger.info("MD finished.")

        # Remove PAUSE sentinel if present
        pause = Path("PAUSE")
        if pause.exists():
            pause.unlink()

        with open("md_out.time", "a") as f:
            f.write(f"{time.time() - initial_time}\n")

    @staticmethod
    def _build_thermostat(name: str, atoms, timestep, temperature, friction, params):
        """Return the appropriate ASE MD integrator."""
        from ase import units

        logfile = "md_out.log"
        if name == "Langevin":
            from ase.md.langevin import Langevin
            return Langevin(atoms, timestep, temperature_K=temperature,
                            friction=friction, logfile=logfile)
        if name == "NoseHooverChain":
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
            return NoseHooverChainNVT(atoms, timestep, temperature_K=temperature,
                                      tdamp=50 * units.fs, tchain=4, logfile=logfile)
        if name == "Berendsen":
            from ase.md.berendsen import Berendsen
            tau = params.get("tau", 25 * timestep)
            return Berendsen(atoms, timestep, temperature_K=temperature,
                             taut=tau, logfile=logfile)
        if name == "VelocityVerlet":
            from ase.md.velocityverlet import VelocityVerlet
            return VelocityVerlet(atoms, timestep, logfile=logfile)
        raise ValueError(f"Unknown thermostat: '{name}'")

    # ------------------------------------------------------------------
    # Minimisation
    # ------------------------------------------------------------------

    @staticmethod
    def run_minimization(
        model_path: str | Path,
        md_params: str | Path,
        type: str = "local",
    ) -> None:
        """
        Run BFGS geometry minimisation using ASE + MACE.

        Parameters
        ----------
        model_path : str or Path
            Path to the trained MACE model.
        md_params : str or Path
            JSON file with keys: ``cell``, ``max_steps`` (int, default 1000),
            ``fmax`` (float, default 0.01 eV/Å), ``interval`` (int, default 1).
        type : {"local"}
            Only "local" is currently implemented.
        """
        from ase.io import read
        from ase.optimize import BFGS
        from mace.calculators import MACECalculator

        if type != "local":
            raise NotImplementedError(f"Minimisation type '{type}' is not yet implemented.")

        model_path = Path(model_path)
        with open(md_params) as f:
            params = json.load(f)

        init_conf = read("coord.xyz", 0)

        if "cell" in params:
            init_conf.cell = np.diag(np.array(params["cell"], dtype=float))
            init_conf.center()
            init_conf.set_pbc(True)
        elif init_conf.cell is not None and init_conf.cell.any():
            init_conf.set_pbc(True)
        else:
            raise ValueError("Cell parameters missing.")

        init_conf.calc = MACECalculator(model_paths=model_path, device="cuda")

        max_steps = int(params.get("max_steps", 1000))
        fmax      = float(params.get("fmax", 0.01))
        interval  = int(params.get("interval", 1))

        dyn = BFGS(init_conf, trajectory="minimization.traj")
        dyn.attach(lambda: dyn.atoms.write("md_out.xyz", append=True), interval=interval)
        dyn.run(fmax=fmax, steps=max_steps)
        logger.info("Minimisation finished.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        model_path: str | Path,
        input_file: str | Path,
        output_file: str | Path,
        type: str = "local",
    ) -> list[tuple[float, np.ndarray]]:
        """
        Run a trained MACE model on all structures in ``input_file`` and write
        predicted energies and forces to ``output_file``.

        Parameters
        ----------
        model_path : str or Path
        input_file : str or Path
            Extended XYZ file with one or more structures.
        output_file : str or Path
            Output extended XYZ with predictions stored in info/arrays.
        type : {"local"}

        Returns
        -------
        list of (energy, forces) tuples for each structure.
        """
        from ase.io import read, write
        from mace.calculators import MACECalculator

        if type != "local":
            raise NotImplementedError(f"Evaluation type '{type}' is not yet implemented.")

        model_path = Path(model_path)
        calculator = MACECalculator(model_paths=model_path, device="cuda")
        structures = read(str(input_file), ":")

        results: list[tuple[float, np.ndarray]] = []
        new_structures = []

        for i, atoms in enumerate(structures):
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            if len(atoms) != forces.shape[0]:
                raise ValueError(
                    f"Structure {i}: atom count ({len(atoms)}) ≠ force rows ({forces.shape[0]})."
                )

            atoms.info["predicted_energy"]   = energy
            atoms.arrays["predicted_forces"] = forces
            new_structures.append(atoms.copy())
            results.append((energy, forces))
            logger.debug(f"Structure {i}: energy = {energy:.4f} eV")

        write(str(output_file), new_structures)
        logger.info(f"Wrote {len(new_structures)} structures → {output_file}")
        return results


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_mlp(method: str, *args, **kwargs) -> MLP:
    """
    Instantiate an MLP wrapper by name.

    Parameters
    ----------
    method : {"mace"}
        The MLP type to create.
    *args, **kwargs :
        Passed to the implementation's ``__init__``.
    """
    if method.lower() == "mace":
        return MACE_MLP(*args, **kwargs)
    raise ValueError(f"Unknown MLP method: '{method}'. Available: ['mace']")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="MACE MLP utilities.")
    p.add_argument(
        "--method",
        choices=["run_md", "run_minimization", "evaluate"],
        required=True,
    )
    args, remaining = p.parse_known_args()

    fn = getattr(MACE_MLP, args.method, None)
    if fn is None:
        logger.error(f"Unknown method: {args.method}")
        sys.exit(1)

    sig = inspect.signature(fn)
    p2 = argparse.ArgumentParser()
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            p2.add_argument(f"--{name}", required=True)
        else:
            p2.add_argument(f"--{name}", default=param.default)

    fn(**vars(p2.parse_args(remaining)))
