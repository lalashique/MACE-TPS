"""Ported verbatim (module-path fixes only) from the original
mlp_path/AIMMD_utils.py.

Not yet wired into a TPSSampler implementation - `ops_backend.py` covers
plain two-way-shooting TPS today. Wiring `AIMMD_setup` into an
`AIMMDTPSSampler(TPSSampler)` (using this to build the RCModel/selector and
swapping the UniformSelector for `setup_selector()`'s RCModel-based one in
`legacy_ops.Sampler.run`) is the natural next step for whoever owns the
TPS/AIMMD package.
"""

import numpy as np
import openpathsampling as paths
from pathlib import Path
import torch
from copy import deepcopy
from util.parser import read_config, print_config
import aimmd


class AIMMD_setup:
    def __init__(self, config_path, descriptor_dim, states, descriptor_transform=None, print_conf=False):
        self.config = read_config(config_path)
        self.system_name = self.config["system_name"]
        self.AIMMD_settings = self.config.get("AIMMD_settings", {})
        self.distribution = self.AIMMD_settings["distribution"]
        self.lorentzian_scale = self.AIMMD_settings["scale"]
        self.train_decision_params = self.AIMMD_settings.get("ee_params", {})
        self.use_GPU = self.AIMMD_settings["use_GPU"]
        self.descriptor_dim = descriptor_dim
        self.descriptor_transform = descriptor_transform
        self.states = states
        if print_conf:
            print_config(self.config)

    def __select_activation(self, in_features=None):
        activation_setting = self.AIMMD_settings.get("activation", "ReLU")
        use_batchnorm = self.AIMMD_settings.get("use_batchnorm", False)
        use_layernorm = self.AIMMD_settings.get("use_layernorm", False)
        activation_params = self.AIMMD_settings.get("activation_params", {})

        activation_map = {
            "ReLU": torch.nn.ReLU(**activation_params),
            "LeakyReLU": torch.nn.LeakyReLU(**activation_params),
            "PReLU": torch.nn.PReLU(**activation_params),
            "ELU": torch.nn.ELU(**activation_params),
            "SELU": torch.nn.SELU(),
            "GELU": torch.nn.GELU(),
            "Tanh": torch.nn.Tanh(),
            "Sigmoid": torch.nn.Sigmoid(),
            "Softplus": torch.nn.Softplus(**activation_params),
            "Softsign": torch.nn.Softsign(),
            "SiLU": torch.nn.SiLU(),
            "Mish": torch.nn.Mish(),
            "Identity": torch.nn.Identity(),
        }
        if activation_setting not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_setting}")

        layers = [activation_map[activation_setting]]
        if use_layernorm:
            if in_features is None:
                raise ValueError("in_features must be provided if using LayerNorm.")
            layers.append(torch.nn.LayerNorm(in_features))
        elif use_batchnorm:
            if in_features is None:
                raise ValueError("in_features must be provided if using BatchNorm.")
            layers.append(torch.nn.BatchNorm1d(in_features))
        return torch.nn.Sequential(*layers)

    def __setup_torch_model(self):
        layers_settings = self.AIMMD_settings.get("layers", {})
        dropout_settings = self.AIMMD_settings.get("dropout", {})
        hidden_layers = [layers_settings[k] for k in layers_settings]
        dropout = [dropout_settings[k] for k in dropout_settings]
        n_unit_layers = [self.descriptor_dim] + hidden_layers

        modules = []
        for i in range(len(hidden_layers)):
            activation = self.__select_activation(in_features=n_unit_layers[i + 1])
            modules.append(
                aimmd.pytorch.networks.FFNet(
                    n_in=n_unit_layers[i],
                    n_hidden=[n_unit_layers[i + 1]],
                    activation=activation,
                    dropout={"0": dropout[i]},
                )
            )
        torch_model = aimmd.pytorch.networks.ModuleStack(n_out=1, modules=modules)

        if torch.cuda.is_available() and self.use_GPU:
            torch_model = torch_model.to("cuda")
        elif torch.backends.mps.is_available() and self.use_GPU:
            torch_model = torch_model.to("mps")
        else:
            torch_model = torch_model.to("cpu")
        return torch_model

    def __setup_descriptor_transform(self):
        return paths.FunctionCV(
            "descriptor_transform", lambda s: s.coordinates[0], cv_wrap_numpy_array=True
        ).with_diskcache()

    def setup_RCModel(self, aimmd_storage, load_model_path=None, loss=None):
        trainset = None
        if load_model_path is not None:
            aimmd_store_old = aimmd.Storage(load_model_path, "r")
            rc_model_old = aimmd_store_old.rcmodels["most_recent"]
            try:
                trainset = deepcopy(aimmd_store_old.load_trainset())
            except Exception:
                trainset = None
            finally:
                torch_model = deepcopy(rc_model_old.nnet)
                if torch.cuda.is_available() and self.use_GPU:
                    torch_model = torch_model.to("cuda")
                elif torch.backends.mps.is_available() and self.use_GPU:
                    torch_model = torch_model.to("mps")
                else:
                    torch_model = torch_model.to("cpu")
                aimmd_store_old.close()
        else:
            torch_model = self.__setup_torch_model()

        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=self.train_decision_params["lr_0"])
        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()

        model = aimmd.pytorch.TIS_EEScalePytorchRCModel(
            nnet=torch_model,
            optimizer=optimizer,
            states=self.states,
            ee_params=self.train_decision_params,
            descriptor_transform=self.descriptor_transform,
            loss=loss,
            cache_file=aimmd_storage,
        )
        aimmd_storage.rcmodels["most_recent"] = model
        if trainset is not None:
            aimmd_storage.save_trainset(trainset)
        return model

    def load_RCModel(self, aimmd_storage, key="most_recent", mode="r"):
        aimmd_store = aimmd.Storage(aimmd_storage, mode)
        model = aimmd_store.rcmodels[key]
        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()
        model.descriptor_transform = self.descriptor_transform

        if torch.cuda.is_available() and self.use_GPU:
            model = model_to(model, "cuda")
        elif torch.backends.mps.is_available() and self.use_GPU:
            model = model_to(model, "mps")
        else:
            model = model_to(model, "cpu")
        return model

    def setup_selector(self, RCModel):
        return aimmd.ops.UniformRCModelSelector(
            model=RCModel,
            states=self.states,
            distribution=self.distribution,
            density_adaptation=True,
            scale=self.lorentzian_scale,
        )


def model_to(model, mode="cuda"):
    model.nnet = model.nnet.to(mode)
    model._device = mode
    model.device = mode
    return model
