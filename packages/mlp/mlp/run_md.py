"""ASE + PLUMED steered-MD runner (port of legacy mlp.py `run_md`).

Run inside an SMD working dir holding `coord.xyz`, the md-params json and (optionally)
`plumed.dat`:

    python -m mlp.run_md --model <path | mace_mp:medium> --md_params smd.json [--plumed plumed.dat]

Outputs (same names as legacy): md_out.xyz, md_out_small.xyz, md_out.log, md_out.time, COLVAR (PLUMED).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mlp.foundation import resolve_model_spec


def _device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_md(model: str, md_params: str | Path, plumed: str | Path | None = None, device: str | None = None) -> None:
    from ase import units
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    params = json.loads(Path(md_params).read_text())
    restart = bool(params.get("restart", False))
    t0 = time.time()

    if restart:
        if not Path("md_out.xyz").exists():
            raise FileNotFoundError("md_out.xyz not found for restart.")
        atoms = read("md_out.xyz", -1)
    else:
        atoms = read("coord.xyz", 0)
    if "cell" in params:
        atoms.cell = np.diag(np.array(params["cell"], dtype=float))
        atoms.center()
        atoms.set_pbc((True, True, True))
    elif atoms.cell.any():
        atoms.set_pbc((True, True, True))
    else:
        raise ValueError("Cell missing in both coord.xyz and md_params.")

    calculator = resolve_model_spec(model).calculator(device=_device(device))

    timestep = float(params.get("time_step", 1.0)) * units.fs
    if plumed:
        from ase.calculators.plumed import Plumed

        plumed = Path(plumed)
        if not plumed.exists():
            raise FileNotFoundError(f"Plumed file not found: {plumed}")
        if "coord.pdb" in plumed.read_text() and not Path("coord.pdb").exists():
            from ase.io import write

            write("coord.pdb", atoms)  # MOLINFO STRUCTURE (names/indices for @mda selectors)
        ps = 1000 * units.fs
        setup = [f"UNITS LENGTH=A TIME={1 / ps} ENERGY={units.mol / units.kJ}"]
        if restart:
            setup.append("RESTART")
        setup.append(f"INCLUDE FILE={plumed}")
        atoms.calc = Plumed(
            calc=calculator, input=setup, timestep=timestep / units.fs / 1000, atoms=atoms, kT=0.03
        )
    else:
        atoms.calc = calculator

    temperature = params.get("temperature", 330)
    thermostat = params.get("thermostat", "Langevin")
    if thermostat == "Langevin":
        dyn = Langevin(
            atoms, timestep, temperature_K=temperature, friction=params.get("friction", 5e-3), logfile="md_out.log"
        )
    elif thermostat == "VelocityVerlet":
        from ase.md.verlet import VelocityVerlet

        dyn = VelocityVerlet(atoms, timestep, logfile="md_out.log")
    else:
        raise ValueError(f"Unknown thermostat: {thermostat}")

    if params.get("initialize_velocities", True) and not restart:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    dyn.attach(lambda: dyn.atoms.write("md_out.xyz", append=True), interval=int(params.get("interval", 1)))
    dyn.attach(
        lambda: dyn.atoms.write("md_out_small.xyz", append=True), interval=int(params.get("small_interval", 50))
    )
    last = [time.time()]

    def _tick():
        with open("md_out.time", "a") as f:
            f.write(f"{time.time() - last[0]}\n")
        last[0] = time.time()

    dyn.attach(_tick, interval=1)
    dyn.run(int(params.get("total_steps", 15000)))
    with open("md_out.time", "a") as f:
        f.write(f"{time.time() - t0}\n")
    print("MD finished!")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", "--model_path", dest="model", required=True)
    p.add_argument("--md_params", required=True)
    p.add_argument("--plumed", default=None)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    run_md(a.model, a.md_params, a.plumed, a.device)


if __name__ == "__main__":
    main()
