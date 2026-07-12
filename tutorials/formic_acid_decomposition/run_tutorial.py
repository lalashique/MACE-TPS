"""Run the formic acid decomposition tutorial: real PySCF DFT labeling,
real MACE committee training, real OpenPathSampling TPS, extension, and
committee (QBC) frame selection, across several generations.

Requires the OpenPathSampling stack (see the main workbench README for how
that was installed into an isolated conda env - PySCF and MACE alone are
plain pip installs and don't need it).

Run: python run_tutorial.py [channel]
  channel: "" (default) for the top-level top.pdb/tps.xyz/tps.json/
           initial_structures.xyz, or a subdirectory name (e.g.
           "dehydration") for a specific channel's seed - see
           setup_system.py, which writes any second-discovered channel to
           its own subdirectory rather than overwriting the first.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # mace-tps-workbench/
for pkg in ("core", "dft", "mlp", "tps", "workflow"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

from core.schemas import WorkflowConfig
from core.state import GenerationStage
from workflow.engine import WorkflowEngine

channel = sys.argv[1] if len(sys.argv) > 1 else ""
SYSTEM_DIR = (HERE / channel) if channel else HERE
run_dir_name = f"run_{channel}" if channel else "run"

tps_template = json.loads((SYSTEM_DIR / "tps.json").read_text())
tps_template["top_file"] = str((SYSTEM_DIR / "top.pdb").resolve())

config = WorkflowConfig(
    root_dir=HERE / run_dir_name,
    initial_structures=SYSTEM_DIR / "initial_structures.xyz",
    initial_trajectory=(SYSTEM_DIR / "tps.xyz").resolve(),
    dft_backend="pyscf",
    mlp_backend="mace",
    tps_backend="ops",
    dft_settings={"basis": "6-31g", "xc": "b3lyp", "charge": 0, "spin": 0},
    mlp_settings={
        # Isolated-atom reference energies at the same level of theory
        # (B3LYP/6-31G, UHF for the open-shell atoms) as the DFT labeling -
        # without these, MACE would be fitting raw PySCF total energies
        # (thousands of eV, dominated by chemically-irrelevant core
        # electrons) instead of the small (tens-of-eV) atomization-energy
        # variations that actually distinguish reactant from products.
        "E0s": {1: -13.613115918006304, 6: -1029.778516874583, 8: -2042.4412224286082},
        "config_file": str(HERE / "mace_train_config.yaml"),
        "valid_fraction": 0.2,
    },
    tps_settings={"config_template": tps_template},
    committee_seeds=[123, 234, 345, 456],
    n_tps_trajectories=40,
    top_k_frames_per_generation=10,
    max_generations=4,
    gates={"dft_submit": "auto", "smd_run": "auto", "tps_run": "auto", "continue_next_gen": "auto"},
)

engine = WorkflowEngine(config)
engine.run_to_completion()

assert engine.is_done()
for record in engine.state.generations:
    print(f"gen {record.index}: stage={record.stage}, blocked={record.blocked}, error={record.error}")
    assert record.stage == GenerationStage.COMPLETE, (record.stage, record.error)
    print(f"  model_paths={record.model_paths}")
    print(f"  candidate_frames_file={record.candidate_frames_file}")
    print(f"  selected_frames_file={record.selected_frames_file}")

print("\nTUTORIAL_PIPELINE_DONE")
