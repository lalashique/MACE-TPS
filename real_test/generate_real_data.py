"""Generate small, real (ASE EMT-calculated) labeled Cu cluster data to feed
into the real MACE backend - thematically close to the paper's Cu system,
but tiny/fast (EMT instead of DFT) so training finishes in well under a
minute on a laptop CPU.
"""

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
import numpy as np

rng = np.random.default_rng(0)

frames = []
base = bulk("Cu", "fcc", a=3.6, cubic=True)  # 4-atom Cu cell
for i in range(12):
    atoms = base.copy()
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.info["energy_cp2k"] = float(energy)
    atoms.arrays["forces_cp2k"] = forces
    frames.append(atoms)

write("real_test/train.xyz", frames[:9], format="extxyz")
write("real_test/test.xyz", frames[9:], format="extxyz")
write("real_test/candidates.xyz", frames, format="extxyz")
print(f"Wrote {len(frames)} EMT-labeled Cu frames (9 train / 3 test / 12 candidates)")
