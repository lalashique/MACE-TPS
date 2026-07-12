import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for pkg in ("core", "mlp", "tps"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

os.chdir(ROOT / "real_test" / "tps_real")

from tps.legacy_ops import TPS_setup, Sampler, extend_and_select_frames

setup = TPS_setup("tps.json")
sampler = Sampler(
    setup_class=setup,
    directory=".",
    output_path="output",
    ops_storage_path="tps_storage.db",
    final_traj_storage_path="traj",
)

config = json.loads(Path("tps.json").read_text())
mace_model_paths = config["mace_model_path"]

print("=== extending trajectories + committee (QBC) uncertainty scoring + weighted selection ===")
selected_path = extend_and_select_frames(sampler=sampler, mace_model_paths=mace_model_paths, workdir=Path("."))
print(f"\nSelected frames written to: {selected_path}")

from ase.io import read

frames = read(selected_path, index=":")
print(f"Selected {len(frames)} frames for next-generation DFT labeling:")
for i, atoms in enumerate(frames):
    print(
        f"  frame {i}: r={atoms.get_distance(0, 1):.3f} A, "
        f"energy_variance={atoms.info.get('energy_variance')}, "
        f"force_variance={atoms.info.get('force_variance')}"
    )

print("\nEXTEND_SELECT_DONE")
