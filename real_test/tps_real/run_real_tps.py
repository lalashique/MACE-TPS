import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # mace-tps-workbench/
for pkg in ("core", "mlp", "tps"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

os.chdir(ROOT / "real_test" / "tps_real")

from tps.legacy_ops import TPS_setup, Sampler

print("=== setting up real OpenMM + MACE system ===")
setup = TPS_setup("tps.json")
print("Setup complete. md_engine:", setup.md_engine)

print("\n=== running real two-way-shooting TPS (OpenPathSampling) ===")
sampler = Sampler(
    setup_class=setup,
    directory=".",
    output_path="output",
    ops_storage_path="tps_storage.db",
    final_traj_storage_path="traj",
)
sampler.run()
print("\nRUN_REAL_TPS_DONE")
