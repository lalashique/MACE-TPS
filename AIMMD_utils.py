import numpy as np
import matplotlib.pyplot as plt
import openpathsampling as paths
from pathlib import Path
import torch
from copy import deepcopy
# Monkey patch OpenPathSampling to use experimental features
from TPS_util import read_config, print_config
import aimmd

class AIMMD_setup:
    def __init__(self, config_path, descriptor_dim,  states, descriptor_transform=None, print_conf=False):
        self.config = read_config(config_path)
        self.system_name =self.config["system_name"]
        self.AIMMD_settings =self.config.get("AIMMD_settings",{})
        self.distribution = self.AIMMD_settings["distribution"]
        self.lorentzian_scale = self.AIMMD_settings["scale"]
        self.train_decision_params = self.AIMMD_settings.get("ee_params",{})
        self.use_GPU = self.AIMMD_settings["use_GPU"]
        self.descriptor_dim = descriptor_dim
        self.descriptor_transform = descriptor_transform
        self.states = states
        if print_conf:
            print_config(self.config)

    def __select_activation(self, in_features=None):
        """
        Returns a nn.Sequential() with activation (and optional BatchNorm) layer based on AIMMD_settings.
        """
        activation_setting = self.AIMMD_settings.get("activation", "ReLU")
        use_batchnorm = self.AIMMD_settings.get("use_batchnorm", False)  
        use_layernorm = self.AIMMD_settings.get("use_layernorm", False) 

        # Optional activation parameters from settings
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
            "SiLU": torch.nn.SiLU(),  # Swish
            "Mish": torch.nn.Mish(),
            "Identity": torch.nn.Identity(),
        }

        if activation_setting not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_setting}")

        layers = [activation_map[activation_setting]]

        # Normalization
        if use_layernorm:   # << prefer this
            if in_features is None:
                raise ValueError("in_features must be provided if using LayerNorm.")
            layers.append(torch.nn.LayerNorm(in_features,))
        elif use_batchnorm: # legacy option
            if in_features is None:
                raise ValueError("in_features must be provided if using BatchNorm.")
            layers.append(torch.nn.BatchNorm1d(in_features))

        return torch.nn.Sequential(*layers)


    def __setup_torch_model(self):

        """ 
        Create the NN model for committor learning and shooting point predictions.
        (In this case the ResNet layer is removed increasing the nonlinearity in the output)

        """

        layers_settings = self.AIMMD_settings.get("layers",{})
        dropout_settings = self.AIMMD_settings.get("dropout", {})
        Hidden_layers = []
        for layer in list(layers_settings.keys()):
            Hidden_layers.append(layers_settings[layer])


        dropout = []
        for layer in list(dropout_settings.keys()):
            dropout.append(dropout_settings[layer])

        n_unit_layers = [self.descriptor_dim] + Hidden_layers

        modules = []
        for i in range(len(Hidden_layers)):
            activation = self.__select_activation(in_features=n_unit_layers[i+1])
            modules += [aimmd.pytorch.networks.FFNet(
                n_in=n_unit_layers[i],
                n_hidden=[n_unit_layers[i+1]],
                activation=activation,
                dropout={"0": dropout[i]}
            )]


        # this step below does an internal linear layer from the final hidden layer size to the final n_out=1 output.
        torch_model = aimmd.pytorch.networks.ModuleStack(n_out=1,
                                                        modules = modules)

        # move model to GPU if CUDA is available
        if torch.cuda.is_available() and self.use_GPU:
            torch_model = torch_model.to('cuda')
            print("using cuda")
        elif torch.backends.mps.is_available() and self.use_GPU:
            torch_model = torch_model.to('mps')
        else: 
            torch_model = torch_model.to("cpu")
        
        return torch_model

    
    
    def __setup_descriptor_transform(self):
        return paths.FunctionCV('descriptor_transform', lambda s: s.coordinates[0], cv_wrap_numpy_array=True).with_diskcache()


    def setup_RCModel(self, aimmd_storage, load_model_path=None, loss=None):
        trainset = None
        if load_model_path is not None:
            # load aimmd storage from model
            aimmd_store_old = aimmd.Storage(load_model_path ,"r")
            RCModel_old = aimmd_store_old.rcmodels["most_recent"]
            try:
                trainset = deepcopy(aimmd_store_old.load_trainset())
            except:
                trainset = None
            finally:
                print("Let me check I should be printing trainset here:", trainset)
                torch_model = deepcopy(RCModel_old.nnet)

                if torch.cuda.is_available() and self.use_GPU:
                    torch_model = torch_model.to('cuda')
                    print("using cuda")
                elif torch.backends.mps.is_available() and self.use_GPU:
                    torch_model = torch_model.to('mps')
                else: 
                    torch_model = torch_model.to("cpu")
                
                aimmd_store_old.close()
            
        else :
            torch_model = self.__setup_torch_model()
                    
        # choose and initialize an optimizer to train the model
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=self.train_decision_params["lr_0"])

        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()

        model = aimmd.pytorch.TIS_EEScalePytorchRCModel(nnet=torch_model,
                                                    optimizer=optimizer,
                                                    states=self.states,
                                                    ee_params=self.train_decision_params,
                                                    descriptor_transform=self.descriptor_transform,  # the function transforming snapshots to descriptors
                                                    loss=loss,  # if loss=None it will choose either binomial or multinomial loss, depending on the number of model outputs,
                                                                # but we could have also passed a custom loss function if we wanted to
                                                    
                                                    cache_file=aimmd_storage,  # cache file for density collector
                                                    )
        aimmd_storage.rcmodels["most_recent"] = model
        if trainset is not None:
            aimmd_storage.save_trainset(trainset)
        return model
    
    def load_RCModel(self,aimmd_storage, key="most_recent", mode="r"):
        # Ensure the file has a .h5 extension using pathlib's with_suffix
        aimmd_store = aimmd.Storage(aimmd_storage, mode)
        model = aimmd_store.rcmodels[key]
        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()
        model.descriptor_transform = self.descriptor_transform
        # move model to GPU if CUDA is available

        use_cuda = self.use_GPU
        #TODO can be single lines with model.nnet = model_cuda object
        if torch.cuda.is_available() and use_cuda:
            model = model_to(model, "cuda")
            print("using cuda")
        elif torch.backends.mps.is_available() and use_cuda:
            model = model_to(model, "mps")
        else: 
            model = model_to(model, "cpu")
        return model

    def setup_selector(self, RCModel):
        selector = aimmd.ops.UniformRCModelSelector(model=RCModel,  # always takes a RCModel
                                     # we can greatly speed up rejecting/accepting trial TPS by passing the list of states
                                     # this enables testing if a trial TP is even a TP and calculating the potentially costly
                                     # transformation from Cartesian to descriptor space only if neccessary
                                     # if we are lazy and know that the transformation is fast we can also explicitly pass None
                                     states=self.states,
                                     # new shooting points are selected with p_sel(SP) ~ p_lorentz(model.z_sel(SP))
                                     # could also choose 'gaussian'
                                     distribution=self.distribution,
                                     density_adaptation=True,
                                     # softness of the selection distribtion,
                                     # lower values result in a sharper concentration around the predicted transition state,
                                     # higher values result in a more uniform selection
                                     scale=self.lorentzian_scale,
                                    )
        return selector
    

def model_to(model,mode="cuda"):
    model.nnet = model.nnet.to(mode)
    model._device = mode
    model.device= mode 
    return model
