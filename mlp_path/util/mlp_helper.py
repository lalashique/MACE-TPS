# committee running to get error estimates
# rmsd from dft data
import numpy as np

# --- PER FRAME PROCESS ---
def process_frame(atoms, carbon_indices, cutoff):
    try:
        energy_var = atoms.info.get("energy_variance")
    except KeyError:
        raise KeyError("energy_variance not found in atoms.info")
    try:
        force_var = atoms.info.get("force_variance")
    except KeyError:
        raise KeyError("force_variance not found in atoms.info")
    try:
        atom_var = atoms.arrays.get("force_var_per_atom")
    except KeyError:
        raise KeyError("force_var_per_atom not found in atoms.arrays")


    pos = atoms.positions
    d_all = np.min([np.linalg.norm(pos - pos[i], axis=1) for i in carbon_indices], axis=0)
    local_idx = np.where(d_all < cutoff)[0]

    if len(local_idx) > 0:
        local_mean = np.mean(atom_var[local_idx])
        local_max = np.max(atom_var[local_idx])
    else:
        local_mean = np.nan
        local_max = np.nan

    return energy_var, force_var, local_mean, local_max

