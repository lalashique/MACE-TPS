"""
util/parser.py — Config file reading and CLI argument construction.

Provides:
- ``read_config``    — load a JSON config file into a dict.
- ``print_config``   — pretty-print a nested config.
- ``create_parser``  — build an argparse.ArgumentParser from a list of
                       argument specs (dict or tuple format).
- ``global_arguments`` — shared argument definitions used across scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def read_config(file_path: str | Path) -> dict:
    """
    Load a JSON configuration file.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If the file contains invalid JSON.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc


def print_config(config: Any, prefix: str = "") -> None:
    """
    Recursively pretty-print a nested config dict or list.

    Parameters
    ----------
    config : dict or list or scalar
    prefix : str
        Indentation prefix (used internally for recursion).
    """
    if isinstance(config, dict):
        for key, value in config.items():
            header = f"{prefix}{key}:"
            if isinstance(value, (dict, list)):
                print(header)
                print_config(value, prefix + "    ")
            else:
                print(f"{header} {value}")
    elif isinstance(config, list):
        for i, item in enumerate(config):
            header = f"{prefix}[{i}]:"
            if isinstance(item, (dict, list)):
                print(header)
                print_config(item, prefix + "    ")
            else:
                print(f"{header} {item}")
    else:
        print(f"{prefix}{config}")


# ---------------------------------------------------------------------------
# Argument parser builder
# ---------------------------------------------------------------------------

def create_parser(arguments_list: list[dict | tuple]) -> argparse.ArgumentParser:
    """
    Build an ``argparse.ArgumentParser`` from a list of argument specs.

    Each spec can be either:

    **dict format** (preferred)::

        {
            "flags":    ["-d", "--directory"],
            "type":     Path,
            "required": True,
            "help":     "Input directory.",
            "default":  None,       # optional
            "nargs":    None,       # optional
            "choices":  None,       # optional
        }

    **tuple format** (legacy)::

        (flags, type, required, help_text[, default[, nargs[, choices]]])

    Parameters
    ----------
    arguments_list : list

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    for spec in arguments_list:
        if isinstance(spec, dict):
            kwargs = {
                "type":     spec.get("type"),
                "required": spec.get("required", False),
                "help":     spec.get("help", ""),
                "default":  spec.get("default"),
                "nargs":    spec.get("nargs"),
            }
            if spec.get("choices"):
                kwargs["choices"] = spec["choices"]
            parser.add_argument(*spec["flags"], **kwargs)

        elif isinstance(spec, tuple):
            flags, arg_type, required, help_text, *rest = spec
            default = rest[0] if len(rest) > 0 else None
            nargs   = rest[1] if len(rest) > 1 else None
            choices = rest[2] if len(rest) > 2 else None
            kwargs = {
                "type":     arg_type,
                "required": required,
                "help":     help_text,
                "default":  default,
            }
            if nargs is not None:
                kwargs["nargs"] = nargs
            if choices is not None:
                kwargs["choices"] = choices
            parser.add_argument(*flags, **kwargs)

        else:
            raise TypeError(f"Unsupported argument spec type: {type(spec)}")

    return parser


# ---------------------------------------------------------------------------
# Shared argument definitions
# ---------------------------------------------------------------------------

global_arguments: dict[str, tuple] = {
    "directory": (
        ["-dir", "--directory"], Path, True,
        "Directory containing the initial trajectory and config files."
    ),
    "config_file": (
        ["-cfg", "--config_file"], str, True,
        "JSON config file with simulation settings."
    ),
    "n_steps": (
        ["-nr", "--n_steps"], int, True,
        "Number of TPS/TIS Monte Carlo cycles."
    ),
    "output_path": (
        ["-out", "--output_path"], Path, False,
        "Directory for storing output files."
    ),
    "previous_model": (
        ["-old_aimmd", "--old_aimmd"], str, False,
        "Path to an existing AIMMD model to continue from."
    ),
    "aimmd_store": (
        ["-aimmd_store", "--aimmd_store"], str, False,
        "Output AIMMD storage filename."
    ),
    "ops_store": (
        ["-ops_store", "--ops_store"], str, False,
        "Output OPS storage filename."
    ),
    "final_storage": (
        ["-final_storage", "--final_storage"], str, False,
        "Save the final trajectory as a separate OPS storage file."
    ),
    "trajectory_path": (
        ["-traj", "--trajectory_path"], Path, False,
        "Path to the input trajectory storage file."
    ),
    "save_final_traj": (
        ["-save_traj", "--save_final_traj"], bool, False,
        "Whether to save the final trajectory as a separate OPS storage.", True
    ),
    "final_traj_storage_path": (
        ["-final_traj_store", "--final_traj_storage_path"], str, False,
        "Path for the final trajectory storage file."
    ),
    "system_resource_directory": (
        ["-sys_source_dir", "--system_resource_directory"], Path, False,
        "Directory for system resource files (e.g. force field files)."
    ),
    "n_epoch": (
        ["-n_epoch", "--n_epoch"], int, True,
        "Number of training epochs."
    ),
    "stable_ops_storages": (
        ["-stable_ops_store", "--stable_ops_store"], str, False,
        "Paths to OPS storage files for stable states A and B.", None, "+"
    ),
    "config_file_TPS": (
        ["-cfg_TPS", "--config_file_TPS"], str, True,
        "JSON config file with TPS simulation settings."
    ),
    "config_file_TIS": (
        ["-cfg_TIS", "--config_file_TIS"], str, True,
        "JSON config file with TIS simulation settings."
    ),
    "interface_model": (
        ["-interface_model", "--interface_model"], str, True,
        "AIMMD model used for interface selection."
    ),
    "interface_values": (
        ["-interface_values", "--interface_values"], float, True,
        "Interface value(s) for TIS.", None, "+"
    ),
    "direction": (
        ["-direction", "--direction"], str, True,
        "Simulation direction.", None, None, ["forward", "backward"]
    ),
    "RPE_name": (
        ["-RPE_name", "--RPE_name"], str, False,
        "RPE storage filename (default: RPE_storage)."
    ),
    "number_trajectories": (
        ["-n_traj", "--n_traj"], int, False,
        "Number of trajectories per TIS run."
    ),
    "n_thermalization": (
        ["-n_therm", "--n_therm"], int, False,
        "Number of trajectories to discard as thermalisation.", 200
    ),
    "cutoff": (
        ["-cutoff", "--cutoff"], float, False,
        "WHAM cutoff.", 0.01
    ),
    "max_min_filename": (
        ["-max_min", "--max_min"], str, False,
        "Filename prefix for the max/min stored path."
    ),
}
