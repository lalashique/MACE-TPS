"""End-to-end smoke test: drives the full DFT -> MLP -> TPS -> select loop
through several generations using ONLY the mock backends, so it needs no
CP2K, no GPU, no real MACE training, no OpenPathSampling.

This is what CI should run on every PR regardless of who touches
dft/mlp/tps - it exercises the workflow engine's state machine and every
backend's interface contract without paying for the real thing.
"""

from pathlib import Path

from core.schemas import WorkflowConfig
from core.state import GenerationStage
from core.xyz_io import Frame, write_frames
from workflow.engine import WorkflowEngine


def _make_toy_system(path: Path, n_frames: int = 5, n_atoms: int = 4):
    import random

    rng = random.Random(42)
    frames = []
    for _ in range(n_frames):
        positions = [
            (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n_atoms)
        ]
        frames.append(Frame(symbols=["C"] * n_atoms, positions=positions))
    write_frames(path, frames)


def test_full_mock_generational_loop(tmp_path):
    root = tmp_path / "run"
    structures = tmp_path / "initial_structures.xyz"
    trajectory = tmp_path / "initial_trajectory.xyz"
    _make_toy_system(structures, n_frames=5)
    _make_toy_system(trajectory, n_frames=10)

    config = WorkflowConfig(
        root_dir=root,
        initial_structures=structures,
        initial_trajectory=trajectory,
        dft_backend="mock",
        mlp_backend="mock",
        tps_backend="mock",
        committee_seeds=[123, 234, 345],
        n_tps_trajectories=8,
        top_k_frames_per_generation=3,
        max_generations=3,
        gates={
            "dft_submit": "auto",
            "smd_run": "auto",
            "tps_run": "auto",
            "continue_next_gen": "auto",
        },
    )

    engine = WorkflowEngine(config)
    engine.run_to_completion()

    assert engine.is_done()
    assert engine.state.current_generation == 3
    assert len(engine.state.generations) == 3

    for record in engine.state.generations:
        assert record.stage == GenerationStage.COMPLETE
        assert not record.blocked
        assert record.error is None
        assert len(record.model_paths) == 3
        assert Path(record.selected_frames_file).exists()

    # state persisted to disk and reloadable
    from core.state import WorkflowState

    reloaded = WorkflowState.load_or_create(root)
    assert reloaded.current_generation == 3


def test_manual_gate_blocks_and_resumes(tmp_path):
    root = tmp_path / "run"
    structures = tmp_path / "initial_structures.xyz"
    trajectory = tmp_path / "initial_trajectory.xyz"
    _make_toy_system(structures, n_frames=5)
    _make_toy_system(trajectory, n_frames=10)

    config = WorkflowConfig(
        root_dir=root,
        initial_structures=structures,
        initial_trajectory=trajectory,
        max_generations=1,
        gates={
            "dft_submit": "manual",
            "smd_run": "auto",
            "tps_run": "auto",
            "continue_next_gen": "auto",
        },
    )

    engine = WorkflowEngine(config)

    from core.gates import WaitingForApproval

    gate_id = None
    try:
        engine.run_to_completion()
        assert False, "expected a WaitingForApproval"
    except WaitingForApproval as e:
        gate_id = e.gate_id
        assert gate_id == "gen0:dft_submit"

    record = engine.state.generation(0)
    assert record.blocked
    assert record.stage == GenerationStage.PENDING

    engine.gates.approve(gate_id)
    engine.run_to_completion()
    assert engine.is_done()
