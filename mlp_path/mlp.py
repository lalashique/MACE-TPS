import sys
import logging
import warnings
from abc import ABC, abstractmethod
from loguru import logger
from pathlib import Path
import yaml
import os

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

# --- Factory helper ---
def create_mlp(method,*args, **kwargs):
    if method.lower() == "mace":
        return MACE_MLP(*args, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    
    mlp = create_mlp("mace", "train.yaml",
                     E0s={1: -13.52638614, 6: -145.7828756, 8: -429.0792406, 29: -1303.851651})
    mlp.train(train_file="train.xyz", test_file="test.xyz", seeds=[123, 234, 345], prepare_only=True,absolute_paths=True)
    # mlp.load("mace_model.model")
    # result = mlp.predict(structure)