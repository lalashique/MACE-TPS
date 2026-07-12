"""Set up everything needed to run the formic acid decomposition tutorial:

  1. Build a formic acid (HCOOH) geometry and relax it with MACE-OFF23
     (a pretrained organic-molecule foundation model - no training needed).
  2. Bootstrap a seed reactive trajectory by running short high-temperature
     MD bursts with that same foundation model until a spontaneous
     decomposition event is observed (dehydration -> CO + H2O, and/or
     dehydrogenation -> CO2 + H2). TPS needs a valid reactant->product path
     handed to it before it can start Monte Carlo sampling of nearby paths -
     this generates that path cheaply, without needing real DFT-based MD or
     hand-built interpolated geometries.
  3. Write top.pdb / tps.xyz / tps.json for the discovered channel(s).

Run: python setup_system.py [--temperature 2500] [--attempts 12]

Atom order (fixed throughout this tutorial):
  O1 (carbonyl oxygen) = 0
  C                     = 1
  O2 (hydroxyl oxygen)  = 2
  Hc (on C)             = 3
  Ho (on O2)            = 4

Requires: ase, mace-torch (pip install ase mace-torch). MACE-OFF23 is
downloaded automatically on first use and cached in ~/.cache/mace/ - it is
licensed under the Academic Software License (non-commercial); it's only
used here to bootstrap a seed trajectory, not distributed with this repo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

torch.set_num_threads(1)

from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase import units

HERE = Path(__file__).resolve().parent
BROKEN_CUTOFF = 2.2  # Angstrom - a bond is considered "broken" past this


class _FoundChannel(Exception):
    pass


def build_and_relax_geometry(calc) -> Atoms:
    positions = {
        "O1": [-0.640, 1.078, 0.0],
        "C": [0.0, 0.0, 0.0],
        "O2": [1.210, -0.350, 0.0],
        "Hc": [-0.930, -0.700, 0.0],
        "Ho": [1.850, 0.400, 0.0],
    }
    atoms = Atoms(
        symbols=["O", "C", "O", "H", "H"],
        positions=[positions[k] for k in ("O1", "C", "O2", "Hc", "Ho")],
    )
    # A small, tight cell (not a large vacuum box) - matscipy's neighbor list
    # (used internally by mace_off) becomes pathologically slow once the box
    # is much bigger than the interaction cutoff (a "Ratio of simulation
    # cell size to cutoff is very large" warning), which only bites once
    # atoms actually drift apart during the high-T search below. This one
    # geometry build is otherwise unaffected by cell size since it's a
    # single relaxation, not open dynamics.
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.center()
    atoms.calc = calc
    BFGS(atoms, logfile=str(HERE / "relax.log")).run(fmax=0.03, steps=200)
    return atoms


def run_md_attempt(relaxed: Atoms, calc, temperature_K: float, seed: int, max_steps: int, tail_steps: int = 300):
    """Run high-T MD until a decomposition channel is detected, then keep
    running `tail_steps` more so the saved path actually settles into the
    product state (a TPS seed needs a real excursion into each state's
    core, not a single frame that just grazes the boundary).

    tail_steps is deliberately short (300, not e.g. 1500): MACE-OFF is an
    approximate potential and can send the system into complete, unphysical
    fragmentation within ~1000 steps of a genuine crossing (observed:
    distances reaching 1e6+ Angstrom) - a longer tail risks capturing that
    instability instead of a clean product-state excursion. The safety guard
    below discards the attempt entirely if that happens anyway.
    """
    atoms = relaxed.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.RandomState(seed))
    dyn = Langevin(atoms, 0.25 * units.fs, temperature_K=temperature_K, friction=0.01)

    frames = []
    result = {"channel": None, "detect_step": None}

    def check():
        step = dyn.nsteps
        d_C_O2 = atoms.get_distance(1, 2)
        d_C_Hc = atoms.get_distance(1, 3)
        d_Hc_O2 = atoms.get_distance(3, 2)
        d_Hc_Ho = atoms.get_distance(3, 4)
        d_O2_Ho = atoms.get_distance(2, 4)

        in_tail = result["detect_step"] is not None
        if in_tail or step % 20 == 0:
            frames.append(atoms.copy())

        if result["channel"] is None:
            if d_C_O2 > BROKEN_CUTOFF and d_Hc_O2 < 1.3:
                result["channel"], result["detect_step"] = "dehydration", step
            elif d_C_Hc > BROKEN_CUTOFF and d_O2_Ho > BROKEN_CUTOFF and d_Hc_Ho < 1.2:
                result["channel"], result["detect_step"] = "dehydrogenation", step
        else:
            if max(d_C_O2, d_C_Hc, d_Hc_O2, d_Hc_Ho, d_O2_Ho) > 15.0:
                # Unphysical fragmentation - discard this attempt rather than
                # save a corrupted seed trajectory.
                result["channel"], result["detect_step"] = None, None
                frames.clear()
                raise _FoundChannel()
            if step - result["detect_step"] >= tail_steps:
                raise _FoundChannel()

    dyn.attach(check, interval=1)
    try:
        dyn.run(max_steps)
    except _FoundChannel:
        pass
    return result["channel"], frames


def write_tps_inputs(seed_frames, channel_name: str):
    """Write top.pdb, tps.xyz, tps.json for the discovered channel directly
    into this tutorial's directory (matching what run_tutorial.py expects).
    If a run finds a second channel, its files are written under a
    subdirectory instead so the first (flat) set isn't overwritten - swap
    run_tutorial.py's paths to point there to explore that channel too.
    """
    is_first = not (HERE / "tps.json").exists()
    out_dir = HERE if is_first else (HERE / channel_name)
    out_dir.mkdir(exist_ok=True)

    first = seed_frames[0].copy()
    first.cell = [10.0, 10.0, 10.0]
    first.pbc = True
    write(str(out_dir / "top.pdb"), first)
    write(str(out_dir / "tps.xyz"), seed_frames, format="xyz")

    config = {
        "system_name": f"formic_acid_{channel_name}",
        "tot_mc_steps": 8,
        "save_every": 4,
        "save_rejected": True,
        "UNITS": {"size_unit": "nanometers", "time_unit": "picoseconds", "temp_unit": "kelvin"},
        "CV": {
            "reaction_progress": {
                "type": "bond",
                "params": {"group_A_atoms": [1, 1], "group_B_atoms": [2, 3], "type": 1},
                "bound": [0.0, 0.16],
                "unbound": [0.22, 1.0],
            }
        },
        "Integrator_settings": {"temperature": 1200, "constant_variable": "T", "dt": 0.0002, "friction": 1.0},
        "mace_model_path": ["PLACEHOLDER_FILLED_IN_BY_WORKFLOW"],
        "platform_name": "CPU",
        "TPS_settings": {"n_frames_max": 6000, "n_steps_per_frame": 1},
        "top_file": str((out_dir / "top.pdb").resolve()),
        "init_path": str((out_dir / "tps.xyz").resolve()),
        "selection_indices": [1, 2, 3],
        "local_range": 5,
        "num_select": 6,
        "extend_steps": 20,
        "alpha": 0.1,
        "local_cutoff": 5.0,
    }
    (out_dir / "tps.json").write_text(json.dumps(config, indent=4))
    print(f"  wrote {out_dir}/{{top.pdb, tps.xyz ({len(seed_frames)} frames), tps.json}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=2500)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=15000)
    args = parser.parse_args()

    from mace.calculators import mace_off

    calc = mace_off(model="small", device="cpu", default_dtype="float32")

    print("=== relaxing formic acid geometry with MACE-OFF23 ===")
    relaxed = build_and_relax_geometry(calc)
    write(str(HERE / "hcooh_relaxed.xyz"), relaxed)
    print(f"  C-O1={relaxed.get_distance(1, 0):.3f}  C-O2={relaxed.get_distance(1, 2):.3f}  "
          f"C-Hc={relaxed.get_distance(1, 3):.3f}  O2-Ho={relaxed.get_distance(2, 4):.3f} A")

    print(f"\n=== searching for decomposition channels (T={args.temperature} K) ===")
    found = {}
    for seed in range(args.attempts):
        channel, frames = run_md_attempt(relaxed, calc, args.temperature, seed, args.max_steps)
        status = channel or "no channel"
        print(f"  attempt {seed}: {status}")
        if channel and channel not in found:
            found[channel] = frames
            write_tps_inputs(frames, channel)
        if len(found) == 2:
            break

    print(f"\nChannels found: {list(found.keys())}")
    if not found:
        raise SystemExit("No channel found - try a higher --temperature or more --attempts.")


if __name__ == "__main__":
    main()
