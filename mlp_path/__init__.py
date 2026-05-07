"""
mlpath — Machine Learning Potential-driven Transition Path Sampling workflow.

Public API
----------
    from mlpath import create_mlp, DFTFactory, TPS_setup, Sampler, Workflow

Modules
-------
- ``dft``      — DFT engine wrappers (CP2K, VASP stub) and factory.
- ``mlp``      — MACE MLP wrapper: training, MD, evaluation, committee.
- ``tps``      — OpenMM + MACE system setup and OPS-based TPS sampler.
- ``workflow`` — Iterative active-learning loop (DFT → MLP → TPS → repeat).
- ``util``     — CV definitions, trajectory utilities, storage helpers.
"""

from .dft      import DFTFactory, DFTBase, CP2KDFT
from .mlp      import MACE_MLP, create_mlp
from .tps      import TPS_setup, Sampler
from .workflow import Workflow

from .util.cv           import CV
from .util.traj_util    import train_test_split, select_frames, split_times
from .util.mlp_helper   import process_frame
from .util.parser       import read_config, print_config, create_parser
from .util.initial_state import ReactionInterface

__all__ = [
    # DFT
    "DFTFactory", "DFTBase", "CP2KDFT",
    # MLP
    "MACE_MLP", "create_mlp",
    # TPS
    "TPS_setup", "Sampler",
    # Workflow
    "Workflow",
    # Utilities
    "CV",
    "train_test_split", "select_frames", "split_times",
    "process_frame",
    "read_config", "print_config", "create_parser",
    "ReactionInterface",
]
