"""Ported from the original mlp_path/mlp.py (MACE_MLP class only - the
run_md/run_minimization/evaluate CLI helpers stayed out of scope for this
pass and can be pulled in later if the MLP owner needs them here too).

Fixed relative to the original:
  - `self._train_func` / `self._calc_cls` were referenced in `train()`/
    `load()` but never assigned (the imports were commented out), so any
    call to `train(prepare_only=False)` or `load()` raised AttributeError.
    They're wired up properly in __init__ now.
"""

from __future__ import annotations

import os
import sys
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
import numpy as np
from loguru import logger

warnings.filterwarnings("ignore")


class MLP(ABC):
    def __init__(self, config_file):
        self.config_file = Path(config_file) if config_file else None
        self.model = None

    @abstractmethod
    def train(self, seed=None, output_dir=None): ...
    @abstractmethod
    def load(self, model_path): ...
    @abstractmethod
    def predict(self, data): ...


class MACE_MLP(MLP):
    DEFAULT_CONFIG = {
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

    def __init__(self, config_file=None, E0s=None):
        super().__init__(config_file)
        self.config = self._get_config()
        self.E0s = E0s
        self.committee_models = []

        from mace.cli.run_train import main as mace_train
        from mace.calculators import MACECalculator

        self._train_func = mace_train
        self._calc_cls = MACECalculator

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
        prepare_only=False,
        absolute_paths=False,
    ):
        """Train one or multiple models (committee)."""
        if seeds is None:
            seeds = [self.DEFAULT_CONFIG["seed"]]
        if not isinstance(seeds, (list, tuple)):
            seeds = [seeds]

        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        trained_models = []
        model_paths = []
        hpc_cmds = []

        for i, seed in enumerate(seeds):
            member_dir = base_dir / (f"committee_{i:02d}" if len(seeds) > 1 else "")
            member_dir, config_name = self._prepare_member(seed, member_dir, train_file, test_file, absolute_paths)
            hpc_cmds.append(f"cd {member_dir} && python -m mace.cli.run_train --config {config_name}")

            if not prepare_only:
                self._run_training(member_dir, config_name)
                # mace.cli.run_train writes the model under model_dir (e.g.
                # "MACE_models/mace01_compiled.model"), not directly in
                # member_dir - a plain member_dir.glob("*.model") (the
                # original code) never finds it.
                model_dir = member_dir / self.config.get("model_dir", "MACE_models")
                model_path = next(model_dir.glob("*.model"), None) or next(
                    member_dir.glob("**/*.model"), None
                )
                if model_path:
                    model_paths.append(model_path)
                    trained_models.append(self._calc_cls(model_paths=str(model_path)))

        if prepare_only:
            return hpc_cmds

        self.committee_models = trained_models
        if len(seeds) == 1 and trained_models:
            self.model = trained_models[0]
        return model_paths

    def _prepare_member(self, seed, output_dir, train_file=None, test_file=None, absolute_paths=False):
        cfg = dict(self.config)
        cfg["seed"] = seed

        if not self.E0s:
            raise ValueError("E0s must be provided for training.")
        cfg["E0s"] = self.E0s

        if train_file:
            if not absolute_paths:
                raise NotImplementedError("Relative paths are not supported (matches original behavior).")
            cfg["train_file"] = str(Path(train_file).absolute())
        if test_file:
            cfg["test_file"] = str(Path(test_file).absolute())

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        config_name = f"train_seed{seed}.yaml"
        tmp_cfg = out / config_name
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

        return out.resolve(), config_name

    def _run_training(self, member_dir, config_name):
        """Run mace.cli.run_train.main() in-process for this committee member.

        The original code built a shell command string ("cd <dir> && python
        -m mace.cli.run_train --config <file>") and fed it to `str.split()`
        to construct sys.argv for an in-process call - argv[0] ended up
        being "cd", which argparse then used as the program name, and the
        --config path was resolved relative to the wrong cwd. That command
        string is only valid for actual shell/HPC execution (still returned
        from train() as the `hpc_cmds`/prepare_only output); running
        in-process needs a real chdir + a real argv instead.
        """
        logging.getLogger().handlers.clear()
        previous_cwd = Path.cwd()
        try:
            os.chdir(member_dir)
            sys.argv = ["run_train", "--config", config_name]
            self._train_func()
        finally:
            os.chdir(previous_cwd)

    def load(self, model_path_or_dir):
        p = Path(model_path_or_dir)
        if p.is_dir():
            models = list(p.glob("*.model"))
            if not models:
                raise FileNotFoundError(f"No model found in {p}")
            p = models[0]
        self.model = self._calc_cls(model_paths=str(p))
        return self.model

    def load_committee(self, model_paths):
        self.committee_models = [self._calc_cls(model_paths=str(p)) for p in model_paths]
        return self.committee_models

    def predict(self, data, committee=False, return_std=False):
        if committee and self.committee_models:
            preds = np.array([m.calculate(data)["energy"] for m in self.committee_models])
            mean = preds.mean(axis=0)
            if return_std:
                return mean, preds.std(axis=0)
            return mean
        elif self.model:
            return self.model.calculate(data)
        raise RuntimeError("No model loaded.")

    @staticmethod
    def evaluate_committee(atoms, committee_calcs):
        """Compute committee energy/force variances and store in atoms.info/arrays."""
        main_calc = atoms.calc
        energies, forces = [], []
        for i, calc in enumerate(committee_calcs):
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
            atoms.info[f"energy_m{i + 1}"] = energies[-1]

        energies = np.array(energies)
        forces = np.stack(forces)
        atoms.info["energy_variance"] = np.var(energies)
        atoms.info["force_variance"] = np.var(forces, axis=0).mean()
        # per-atom force variance - needed by tps/util/mlp_helper.py::process_frame
        # for its local-neighborhood variance calculation. Dropped when this was
        # first ported here; restored to match the original mlp_path/mlp.py.
        atoms.arrays["force_var_per_atom"] = np.var(forces, axis=0).mean(axis=1)
        atoms.calc = main_calc
        return atoms


def create_mlp(method, *args, **kwargs):
    if method.lower() == "mace":
        return MACE_MLP(*args, **kwargs)
    raise ValueError(f"Unknown method: {method}")
