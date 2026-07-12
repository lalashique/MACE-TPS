"""Real-backend smoke test: drives dft.cp2k_backend.CP2KDFTEngine.label()
end-to-end against a real, tiny (3-atom water) CP2K reftraj calculation.

First call: engine has no state yet, prepares CP2K reftraj inputs, and
raises NeedsExternalJob with the exact command to run - this test then runs
that command itself (real CP2K, ~10-15s on a laptop CPU) and calls label()
again to prove the resume-after-external-job path works.
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, "packages/core")
sys.path.insert(0, "packages/dft")

from core.interfaces import NeedsExternalJob
from dft.cp2k_backend import CP2KDFTEngine

structures = Path("real_test/dft_real/coord.xyz")
workdir = Path("real_test/dft_real_run")

engine = CP2KDFTEngine(
    settings={"command": "/opt/homebrew/bin/cp2k.psmp"},
    input_files=["real_test/dft_real/cp2k.inp"],
)

print("=== first call: expect NeedsExternalJob (CP2K hasn't run yet) ===")
try:
    engine.label(structures, workdir)
    raise SystemExit("expected NeedsExternalJob on first call, got none")
except NeedsExternalJob as e:
    print("Got expected NeedsExternalJob:")
    print(e.instructions)

struct_dir = workdir / "structure_0"
print(f"\n=== running real CP2K in {struct_dir} ===")
result = subprocess.run(
    ["/opt/homebrew/bin/cp2k.psmp", "-i", "cp2k.inp", "-o", "cp2k.out"],
    cwd=struct_dir,
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(result.stdout[-3000:])
    print(result.stderr[-3000:])
    raise SystemExit(f"CP2K failed with code {result.returncode}")
print("CP2K finished successfully.")

print("\n=== second call: expect the engine to parse outputs and return the labeled path ===")
labeled_path = engine.label(structures, workdir)
print(f"Labeled file: {labeled_path}")
assert labeled_path.exists()

from ase.io import read

frames = read(labeled_path, index=":")
print(f"Parsed {len(frames)} labeled frames:")
for i, atoms in enumerate(frames):
    print(f"  frame {i}: energy_cp2k={atoms.info['energy_cp2k']:.4f} eV, "
          f"forces_cp2k shape={atoms.arrays['forces_cp2k'].shape}")
assert len(frames) == 3
for atoms in frames:
    assert "energy_cp2k" in atoms.info
    assert atoms.arrays["forces_cp2k"].shape == (3, 3)

print("\nREAL_DFT_TEST_PASSED")
