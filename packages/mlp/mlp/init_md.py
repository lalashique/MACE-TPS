"""Initial-data generation: Langevin MD of one structure with a foundation model (MACE-MP).

Paper (arXiv:2605.03737 SI): 1 ns MD at 400 K (1 fs step), 400 equally spaced frames per structure.
One call handles ONE input xyz (with its box); the workflow engine calls it once per input file.

    python -m mlp.init_md --model mace_mp:medium --input a.xyz --output init_0.xyz --settings init_md.json

settings keys (all optional): temperature (400), time_step fs (1.0), total_steps (1_000_000),
n_frames (400), friction_per_fs (1e-3 == 1/ps, see note), cell ([a,b,c] or 3x3, used only when the xyz has
no cell), seed (0), device ("auto"), thermostat ("Langevin" | "VelocityVerlet").

Note on friction: the paper's SI states "1 fs^-1" without a consistent unit (the TPS config uses 1.0 in OpenMM's
ps^-1). Default here is the physical 1 ps^-1 = 1e-3 fs^-1; override with friction_per_fs.

Resumable: frames already in --output are kept; MD continues from the last frame (velocities are re-drawn).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlp.foundation import resolve_model_spec

DEFAULTS = dict(
    temperature=400.0, time_step=1.0, total_steps=1_000_000, n_frames=400,
    friction_per_fs=1e-3, seed=0, device="auto", thermostat="Langevin",
)


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_init_md(model: str, input_xyz: str | Path, output_xyz: str | Path, settings: dict | None = None) -> int:
    """Run MD, return number of frames in output. Raises if no box is available."""
    from ase import units
    from ase.io import read, write
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    p = {**DEFAULTS, **(settings or {})}
    n_frames, total_steps = int(p["n_frames"]), int(p["total_steps"])
    if n_frames < 1 or total_steps < n_frames:
        raise ValueError(f"need total_steps >= n_frames >= 1 (got {total_steps}, {n_frames})")
    interval = total_steps // n_frames
    output_xyz = Path(output_xyz)

    done = len(read(output_xyz, ":")) if output_xyz.exists() and output_xyz.stat().st_size else 0
    if done >= n_frames:
        return done
    atoms = read(output_xyz, -1) if done else read(input_xyz, 0)
    if not atoms.cell.any():
        if "cell" not in p:
            raise ValueError(f"{input_xyz}: no box (cell) in the xyz and none given in settings['cell']")
        cell = np.array(p["cell"], dtype=float)
        atoms.cell = np.diag(cell) if cell.ndim == 1 else cell
    atoms.set_pbc(True)

    atoms.calc = resolve_model_spec(model).calculator(device=_device(p["device"]))
    dt = float(p["time_step"]) * units.fs
    if p["thermostat"] == "Langevin":
        dyn = Langevin(atoms, dt, temperature_K=float(p["temperature"]),
                       friction=float(p["friction_per_fs"]) / units.fs, rng=np.random.default_rng(int(p["seed"]) + done))
    elif p["thermostat"] == "VelocityVerlet":
        dyn = VelocityVerlet(atoms, dt)
    else:
        raise ValueError(f"Unknown thermostat: {p['thermostat']}")
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(p["temperature"]),
                                 rng=np.random.default_rng(int(p["seed"]) + done))

    count = [done]
    skip_first = [done > 0]  # ASE calls observers at step 0: on resume that frame is already in the file

    def _save():
        if skip_first[0]:
            skip_first[0] = False
            return
        write(output_xyz, atoms, append=True)
        count[0] += 1

    dyn.attach(_save, interval=interval)
    # fresh: frames at steps 0, interval, ... (n_frames-1)*interval; resume: continue after the last frame
    dyn.run(interval * (n_frames - done) if done else interval * (n_frames - 1))
    return count[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--settings", default=None, help="json file with settings")
    a = ap.parse_args(argv)
    settings = json.loads(Path(a.settings).read_text()) if a.settings else {}
    n = run_init_md(a.model, a.input, a.output, settings)
    print(f"init MD finished: {n} frames -> {a.output}")


if __name__ == "__main__":
    main()
