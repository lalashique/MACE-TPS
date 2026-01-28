import os
from pathlib import Path
import numpy as np
from ase.io import read, write  
from loguru import logger

def train_test_split(xyz, test_fraction=0.25):
    """Splits data into training and testing sets."""

    try:
        # save the file
        atoms_list = read(os.path.join(xyz),':')
        # Shuffle and split
        total = len(atoms_list)
        indices = np.arange(total)
        np.random.shuffle(indices)
        train_cut = int((1 - test_fraction) * total)
        train_indices = indices[:train_cut]
        test_indices = indices[train_cut:]

        train_atoms = [atoms_list[i] for i in train_indices]
        test_atoms = [atoms_list[i] for i in test_indices]
        #save to folder containing original file
        folder = os.path.dirname(xyz)
        write(os.path.join(folder, "train.xyz"), train_atoms)
        write(os.path.join(folder, "test.xyz"), test_atoms)
        # print(f"Saved {len(train_atoms)} train and {len(test_atoms)} test samples.")

    except FileNotFoundError:
        print(f"File not found.")
        # continue
    except AssertionError:
        print(f"Assertion error.")
        # continue%

from ase.io import read, write
import numpy as np

def select_frames(
    xyz_path,
    tps_params,
    out_file="selected.xyz",
):
    frames = read(xyz_path, index=":")
    n = len(frames)

    selection_indices = tps_params.get("selection_indices")
    local_range = tps_params.get("local_range")
    num_select = tps_params.get("num_select")

    # extract metrics
    try:
        energy_var = np.array([f.info["energy_variance"] for f in frames])
        force_var  = np.array([f.info["force_variance"] for f in frames])
        fpa        = np.array([f.arrays["force_var_per_atom"] for f in frames])
    except KeyError as e:
        raise(f"KeyError: {e}. Ensure that the frames have the required info and arrays.")

    fpa_mean   = fpa.mean(axis=1)
    fpa_max    = fpa.max(axis=1)

    collected = []

    # 1. energy variance (global)
    worst_energy = np.argsort(energy_var)[-num_select:]
    collected.append(worst_energy)

    # 2. force variance (global)
    worst_force = np.argsort(force_var)[-num_select:]
    collected.append(worst_force)

    # 3. local mean fpa (local windows)
    if selection_indices is not None:
        mean_picks = []
        for idx in selection_indices:
            lo = max(idx - local_range, 0)
            hi = min(idx + local_range + 1, n)
            window_ids = np.arange(lo, hi)

            vals = fpa_mean[window_ids]
            worst_local = window_ids[np.argsort(vals)[-num_select:]]
            mean_picks.append(worst_local)

        mean_picks = np.unique(np.concatenate(mean_picks))
        collected.append(mean_picks)

    # 4. local max fpa (local windows)
    if selection_indices is not None:
        max_picks = []
        for idx in selection_indices:
            lo = max(idx - local_range, 0)
            hi = min(idx + local_range + 1, n)
            window_ids = np.arange(lo, hi)

            vals = fpa_max[window_ids]
            worst_local = window_ids[np.argsort(vals)[-num_select:]]
            max_picks.append(worst_local)

        max_picks = np.unique(np.concatenate(max_picks))
        collected.append(max_picks)

    # merge all
    final_ids = np.unique(np.concatenate(collected))

    # rank by severity
    severity = (
        energy_var[final_ids] +
        force_var[final_ids] +
        fpa_mean[final_ids] +
        fpa_max[final_ids]
    )

    # keep worst 2000
    k = num_select
    if len(final_ids) > k:
        final_ids = final_ids[np.argsort(severity)[-k:]]

    # write
    write(out_file, [frames[i] for i in final_ids])
    print(f"Wrote {len(final_ids)} frames → {out_file}")

def split_times(xyz, split=4):
    """Splits a trajectory file into multiple segments after shuffling."""
    try:
        # read the file
        atoms_list = read(xyz,':')
        np.random.shuffle(atoms_list)
        total = len(atoms_list)
        segment_size = total // split
        base, ext = os.path.splitext(xyz)

        for i in range(split):
            start = i * segment_size
            end = (i + 1) * segment_size if i < split - 1 else total
            segment_atoms = atoms_list[start:end]
            out_file = f"{base}_part{i+1}{ext}"
            write(out_file, segment_atoms)
            logger.info(f"Saved segment {i+1} with {len(segment_atoms)} frames to {out_file}.")

    except FileNotFoundError:
        print(f"File not found: {xyz}")
    except AssertionError:
        print(f"Assertion error while processing the file: {xyz}")