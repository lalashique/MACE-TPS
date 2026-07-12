"""Full real-backend workflow test: drives WorkflowEngine end-to-end with
mock DFT (already proven extxyz-interoperable with real MACE) + REAL MACE
training + REAL OpenPathSampling TPS, through the actual factory-resolved
adapters (mlp.mace_backend.MACEBackend, tps.ops_backend.OPSTPSSampler) - not
just the raw legacy_ops/legacy_mace classes tested directly so far. This is
what exercises workflow/engine.py's generation loop with everything real
except the DFT label step itself.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for pkg in ("core", "dft", "mlp", "tps", "workflow"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

os.chdir(ROOT / "real_test" / "tps_real")

from core.schemas import WorkflowConfig
from core.state import GenerationStage
from core.xyz_io import Frame, write_frames
from workflow.engine import WorkflowEngine

# A handful of diatomic H-H structures spanning bound/barrier/unbound, for
# mock DFT to "label" with its own toy potential each generation (real CP2K
# on this exact synthetic double-well system isn't meaningful - CP2K would
# compute the REAL H2 curve, not the toy double-well MACE was trained on;
# proving mock-DFT-format compatibility with real MACE was already done
# separately in run_mixed_workflow_test.py).
initial_structures = Path("full_workflow_structures.xyz")
write_frames(
    initial_structures,
    [Frame(symbols=["H", "H"], positions=[[0, 0, 0], [r, 0, 0]]) for r in
     [0.9, 1.0, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]],
)

config = WorkflowConfig(
    root_dir=Path("full_workflow_run"),
    initial_structures=initial_structures,
    initial_trajectory=Path("tps.xyz"),
    dft_backend="mock",
    mlp_backend="mace",
    tps_backend="ops",
    mlp_settings={"E0s": {1: 0.0}, "config_file": "mace_train_config.yaml", "valid_fraction": 0.25},
    tps_settings={
        "config_template": {
            "system_name": "diatomic_test",
            "tot_mc_steps": 6,
            "save_every": 3,
            "save_rejected": True,
            "UNITS": {"size_unit": "nanometers", "time_unit": "picoseconds", "temp_unit": "kelvin"},
            "CV": {
                "bond_dist": {
                    "type": "bond",
                    "params": {"group_A_atoms": [0], "group_B_atoms": [1], "type": 1},
                    "bound": [0.0, 0.13],
                    "unbound": [0.22, 0.40],
                }
            },
            "Integrator_settings": {"temperature": 400, "constant_variable": "T", "dt": 0.0005, "friction": 1.0},
            "platform_name": "CPU",
            "TPS_settings": {"n_frames_max": 4000, "n_steps_per_frame": 1},
            "top_file": str(Path("top.pdb").resolve()),
            "selection_indices": [0, 1],
            "local_range": 5,
            "num_select": 8,
            "extend_steps": 10,
            "alpha": 0.1,
            "local_cutoff": 5.0,
        }
    },
    committee_seeds=[123, 234],
    n_tps_trajectories=6,
    top_k_frames_per_generation=8,
    max_generations=2,
    gates={"dft_submit": "auto", "smd_run": "auto", "tps_run": "auto", "continue_next_gen": "auto"},
)

engine = WorkflowEngine(config)
engine.run_to_completion()

assert engine.is_done()
for record in engine.state.generations:
    print(f"gen {record.index}: stage={record.stage}, blocked={record.blocked}, error={record.error}")
    assert record.stage == GenerationStage.COMPLETE, (record.stage, record.error)
    assert not record.blocked
    assert record.error is None
    print(f"  model_paths={record.model_paths}")
    print(f"  candidate_frames_file={record.candidate_frames_file}")
    print(f"  selected_frames_file={record.selected_frames_file}")
    assert Path(record.selected_frames_file).exists()

print("\nFULL_REAL_WORKFLOW_DONE")
