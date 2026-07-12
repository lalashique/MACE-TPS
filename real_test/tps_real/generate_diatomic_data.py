"""Synthetic double-well diatomic PES: a bound well around r0=1.0 Angstrom,
a real energetic barrier around rb=1.6 Angstrom, and an unbound plateau past
~2.4 Angstrom. This gives TPS an actual rare event to sample (bound <->
unbound crossing over a barrier), analogous in spirit to a bond-breaking
reaction coordinate, without needing real DFT-quality chemistry - the point
is to exercise the real TPS/OpenPathSampling + real MACE-as-forcefield
machinery end-to-end.
"""

import numpy as np
from ase import Atoms
from ase.io import write

D = 0.35      # Morse well depth (eV)
A = 2.2       # Morse steepness (1/Angstrom)
R0 = 1.0      # Morse minimum (Angstrom)
B = 0.22      # barrier bump height (eV)
RB = 1.6      # barrier bump location (Angstrom)
W = 0.22      # barrier bump width (Angstrom)


def energy_and_force(r):
    morse = D * ((1 - np.exp(-A * (r - R0))) ** 2 - 1)
    bump = B * np.exp(-((r - RB) / W) ** 2)
    energy = morse + bump

    dmorse_dr = 2 * D * (1 - np.exp(-A * (r - R0))) * (A * np.exp(-A * (r - R0)))
    dbump_dr = -2 * (r - RB) / W**2 * bump
    dE_dr = dmorse_dr + dbump_dr
    return energy, -dE_dr  # force magnitude along the bond


def make_frame(r):
    positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]]
    energy, f_r = energy_and_force(r)
    forces = [[-f_r, 0.0, 0.0], [f_r, 0.0, 0.0]]
    atoms = Atoms(symbols=["H", "H"], positions=positions, cell=[20, 20, 20], pbc=False)
    atoms.info["energy_cp2k"] = float(energy)
    atoms.arrays["forces_cp2k"] = np.array(forces)
    return atoms


if __name__ == "__main__":
    rs = np.concatenate(
        [
            np.linspace(0.75, 1.35, 16),   # dense around the bound well
            np.linspace(1.35, 2.0, 14),    # across the barrier
            np.linspace(2.0, 3.6, 14),     # unbound plateau
        ]
    )
    frames = [make_frame(r) for r in rs]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    frames = [frames[i] for i in order]
    n_test = max(1, len(frames) // 5)

    write("real_test/tps_real/train.xyz", frames[n_test:], format="extxyz")
    write("real_test/tps_real/test.xyz", frames[:n_test], format="extxyz")
    print(f"Wrote {len(frames) - n_test} train / {n_test} test diatomic frames, r in "
          f"[{rs.min():.2f}, {rs.max():.2f}] Angstrom")
