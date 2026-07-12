import numpy as np
from ase import Atoms
from ase.io import write

# top.pdb: single reference frame (bound state), used by ISetup only to
# build the OpenMM topology/box. ISetup._setup_system reads this with
# ase.io.read and explicitly divides by 10 (Angstrom -> nm), so this file
# must be in real Angstrom (standard PDB convention).
top = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [1.0, 0, 0]], cell=[20, 20, 20], pbc=True)
write("real_test/tps_real/top.pdb", top)

# tps.xyz: a crude seed reactive trajectory that actually crosses from the
# bound well (r ~ 1.0 A) over the barrier (~1.6 A) into the unbound region
# (r > 2.4 A) - OPS's two-way shooting needs a valid A->B path to bootstrap
# from, even a deterministic/non-thermal one like this.
#
# Empirically verified (by loading this file with mdtraj directly and
# checking the resulting distances): mdtraj's bare-.xyz reader treats the
# numbers as Angstrom and auto-converts to its native nm internally - NOT
# "already nm" as mdtraj's own docs suggest for ambiguous formats. Writing
# nm-scaled numbers here (an earlier version of this script did exactly
# that) produced trajectories 10x too short once loaded, which silently
# broke OPS's initial_conditions_from_trajectories (no valid A->B sample
# found). So: write real Angstrom here, same as top.pdb.
rng = np.random.default_rng(1)
n_frames = 80
r_values_angstrom = np.linspace(0.95, 3.2, n_frames) + rng.normal(scale=0.01, size=n_frames)

frames = []
for r in r_values_angstrom:
    frames.append(Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [r, 0, 0]], cell=[20, 20, 20], pbc=False))

write("real_test/tps_real/tps.xyz", frames, format="xyz")
print(
    f"Wrote top.pdb and a {n_frames}-frame seed trajectory (both Angstrom) from "
    f"r={r_values_angstrom[0]:.3f} to r={r_values_angstrom[-1]:.3f} A"
)
