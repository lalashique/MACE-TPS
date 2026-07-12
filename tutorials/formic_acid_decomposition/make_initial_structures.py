"""Sample a diverse set of formic acid geometries along the bootstrapped
seed trajectory (bound well, near the barrier, unbound product) to use as
generation-0 DFT labeling targets - the same "approximate barrier-crossing
structures" bootstrap strategy described for the real active-learning loop,
just using the MACE-OFF-bootstrapped path instead of constrained DFT MD.
"""
from ase.io import read, write
import numpy as np

HERE = __file__.rsplit("/", 1)[0]

frames = read(f"{HERE}/tps.xyz", index=":")
n = len(frames)
idx = np.unique(np.linspace(0, n - 1, 40).astype(int))
selected = [frames[i] for i in idx]
write(f"{HERE}/initial_structures.xyz", selected)
print(f"Wrote {len(selected)} initial structures (of {n} seed frames) to initial_structures.xyz")
