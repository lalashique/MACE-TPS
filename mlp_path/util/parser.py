
import argparse
from pathlib import Path
import json

def create_parser(arguments_list):
    """
    Creates an ArgumentParser with arguments specified in arguments_list.

    Args:
        arguments_list (list): List of tuples or dictionaries defining arguments:
            - If tuple: (flags, type, required, help_text, [default], [nargs])
            - If dictionary: {
                "flags": list of flags (e.g., ["-d", "--directory"]),
                "type": argument type (e.g., str, int, Path),
                "required": if the argument is required,
                "help": help message for the argument,
                "default": (optional) default value if argument is not provided,
                "nargs": (optional) number of arguments expected (e.g., "+", "*")
              }

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser
    """
    parser = argparse.ArgumentParser()
    for arg in arguments_list:
        if isinstance(arg, tuple):  # Handle tuple format
            # Unpack flags, type, required, help_text, with optional default and nargs
            flags, arg_type, required, help_text, *rest = arg
            default = rest[0] if len(rest) > 0 and not isinstance(rest[0], str) else None
            nargs = rest[1] if len(rest) > 1 and isinstance(rest[1], str) else None
            
            # Add the argument to the parser
            parser.add_argument(
                *flags,
                type=arg_type,
                required=required,
                help=help_text,
                default=default,
                nargs=nargs
            )
        elif isinstance(arg, dict):  # Handle dictionary format
            # Add the argument to the parser using dict keys
            parser.add_argument(
                *arg["flags"],
                type=arg["type"],
                required=arg.get("required", False),
                help=arg.get("help", ""),
                default=arg.get("default", None),
                nargs=arg.get("nargs", None)
            )
    return parser


def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError as e:
            raise(ValueError(f"Error parsing JSON configuration file: {e}"))
    return config

def print_config(config, prefix=''):
    if isinstance(config, dict):
        for key, value in config.items():
            new_prefix = f"{prefix}{key}: "
            if isinstance(value, (dict, list)):
                print(new_prefix)
                print_config(value, prefix + '    ')
            else:
                print(f"{new_prefix}{value}")
    elif isinstance(config, list):
        for index, item in enumerate(config):
            new_prefix = f"{prefix}[{index}]: "
            if isinstance(item, (dict, list)):
                print(new_prefix)
                print_config(item, prefix + '    ')
            else:
                print(f"{new_prefix}{item}")


# Global argument dictionary
global_arguments = {
    "directory": (["-dir", "--directory"], Path, True, "Directory containing initial trajectory and configuration files."),
    "config_file": (["-cfg", "--config_file"], str, True, "File in python configparser format with simulation settings."),
    "n_steps": (["-nr", "--n_steps"], int, True, "The number of desired TPS/TIS MC cycles."),
    "output_path": (["-out", "--output_path"], Path, False, "Directory for storing TPS output files."),
    "previous_model": (["-old_aimmd", "--old_aimmd"], str, False, "Name of the old AIMMD model to be used."),
    "aimmd_store": (["-aimmd_store", "--aimmd_store"], str, False, "Name for output AIMMD storage, default describes which selector and PES is used."),
    "ops_store": (["-ops_store", "--ops_store"], str, False, "Name for output OPS storage, default describes which selector and PES is used."),
    "final_storage": (["-final_storage", "--final_storage"], str, False, "Save final trajectory as separate OPS storage."),
    "trajectory_path": (["-traj", "--trajectory_path"], Path, False, "Path to the input trajectory storage file."),
    "save_final_traj": (["-save_traj", "--save_final_traj"], bool, False, "Flag to save the final trajectory as a separate OPS storage.", True),
    "final_traj_storage_path": (["-final_traj_store", "--final_traj_storage_path"], str, False, "Path to save the final trajectory storage file."),
    "system_resource_directory": (["-sys_source_dir", "--system_resource_directory"], Path, False, "Directory for system resource files."),
    "n_epoch": (["-n_epoch", "--n_epoch"], int, True, "The number of epochs for training."),
    "stable_ops_storages":(["-stable_ops_store", "--stable_ops_store"], str, False, "Paths to OPS storage files for stable states A and B.", None, "+"),
    "config_file_TPS": (["-cfg_TPS", "--config_file_TPS"], str, True, "File in python configparser format with simulation settings."),
    "config_file_TIS": (["-cfg_TIS", "--config_file_TIS"], str, True, "File in python configparser format with TIS settings."),
    "interface_model": (["-interface_model", "--interface_model"], str, True, "Name of the old AIMMD model to be used."),
    "interface_values": (["-interface_values", "--interface_values"], float, True, "The value of the interface used in the simulation.", None, "+"),
    "direction": (["-direction", "--direction"], str, True, "The direction of the simulation, either 'forward' or 'backward'.", None, None, ["forward", "backward"]),
    "RPE_name": (["-RPE_name","--RPE_name"],str,False,"RPE storage name, default name is RPE_storage"),
    "number_trajectories": (["-n_traj", "--n_traj"], int, False, "The number of trajectories obtained for each TIS run."),
    "n_thermalization": (["-n_therm", "--n_therm"], int, False, "The number of trajectories removed as thermalization.", 200),
    "cutoff": (["-cutoff", "--cutoff"], float, False, "The WHAM cutoff used.", 0.01),
    "max_min_filename": (["-max_min", "--max_min"], str, False, "filename for the max_min stored path (should be first part of the name e.g. 'max_min_q_int')")
}
