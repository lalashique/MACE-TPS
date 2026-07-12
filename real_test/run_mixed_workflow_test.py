"""Real-backend mixed test: drives the FULL WorkflowEngine (mock DFT + mock
TPS + REAL MACE training/committee) across 2 generations. Proves the
"mix and match backends" design in the README actually works, not just the
MLP backend in isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, "packages/core")
sys.path.insert(0, "packages/dft")
sys.path.insert(0, "packages/mlp")
sys.path.insert(0, "packages/tps")
sys.path.insert(0, "packages/workflow")

from core.schemas import WorkflowConfig
from core.state import GenerationStage
from core.xyz_io import Frame, write_frames
from workflow.engine import WorkflowEngine

import random


def make_toy_system(path, n_frames, n_atoms=4, seed=1):
    rng = random.Random(seed)
    frames = []
    for _ in range(n_frames):
        positions = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n_atoms)]
        frames.append(Frame(symbols=["Cu"] * n_atoms, positions=positions))
    write_frames(path, frames)


root = Path("real_test/mixed_run")
structures = Path("real_test/mixed_structures.xyz")
trajectory = Path("real_test/mixed_trajectory.xyz")
make_toy_system(structures, n_frames=10)
make_toy_system(trajectory, n_frames=16, seed=2)

config = WorkflowConfig(
    root_dir=root,
    initial_structures=structures,
    initial_trajectory=trajectory,
    dft_backend="mock",
    mlp_backend="mace",  # <-- REAL backend
    tps_backend="mock",
    mlp_settings={"E0s": {29: 0.0}, "config_file": "real_test/mace_train_config.yaml", "valid_fraction": 0.3},
    committee_seeds=[123, 234],
    n_tps_trajectories=12,
    top_k_frames_per_generation=10,
    max_generations=2,
    gates={"dft_submit": "auto", "smd_run": "auto", "tps_run": "auto", "continue_next_gen": "auto"},
)

engine = WorkflowEngine(config)
engine.run_to_completion()

assert engine.is_done()
for record in engine.state.generations:
    assert record.stage == GenerationStage.COMPLETE, record.stage
    assert not record.blocked
    assert record.error is None, record.error
    assert len(record.model_paths) == 2
    for p in record.model_paths:
        assert Path(p).exists(), f"missing real model file: {p}"
    assert Path(record.selected_frames_file).exists()
    print(f"gen {record.index}: model_paths={record.model_paths}")
    print(f"gen {record.index}: selected_frames_file={record.selected_frames_file}")

print("\nMIXED_WORKFLOW_TEST_PASSED")
