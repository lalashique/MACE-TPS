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


def _cfg(tmp_path, **kw):
    structures = tmp_path / "s.xyz"
    trajectory = tmp_path / "t.xyz"
    _make_toy_system(structures, n_frames=8)
    _make_toy_system(trajectory, n_frames=10)
    gates = {k: "auto" for k in ("dft_submit", "smd_run", "tps_run", "continue_next_gen")}
    base = dict(
        root_dir=tmp_path / "run", initial_structures=structures, initial_trajectory=trajectory,
        n_tps_trajectories=8, top_k_frames_per_generation=6, max_generations=2, gates=gates,
    )
    base.update(kw)
    return WorkflowConfig(**base)


def test_multi_input_split_cumulative_and_sampler_selection(tmp_path):
    from core.xyz_io import iter_raw_blocks

    a, b = tmp_path / "a.xyz", tmp_path / "b.xyz"
    _make_toy_system(a, n_frames=4)
    _make_toy_system(b, n_frames=4)
    config = _cfg(
        tmp_path, initial_structures=[a, b], cumulative_training=True,
        selection_mode="sampler", dft_frames_per_split=4,
        mlp_settings={"valid_fraction": 0.25},
    )
    engine = WorkflowEngine(config)
    engine.run_to_completion()

    g0, g1 = engine.state.generation(0), engine.state.generation(1)
    assert len(iter_raw_blocks(Path(g0.labeled_file))) == 8  # list input concatenated by the DFT backend
    # sampler mode: selected == candidate pool, not top-k
    assert len(iter_raw_blocks(Path(g0.selected_frames_file))) == len(iter_raw_blocks(Path(g0.candidate_frames_file)))
    n1 = len(iter_raw_blocks(Path(g1.labeled_file)))
    # gen 1 was split into ceil(frames/4) chunks and trained cumulatively
    assert (tmp_path / "run" / "1" / "dft" / "split_1.xyz").exists()
    total = len(iter_raw_blocks(Path(g1.train_file))) + len(iter_raw_blocks(Path(g1.test_file)))
    assert total == 8 + n1


def test_smd_gate_blocks_until_result_present(tmp_path):
    from core.gates import WaitingForApproval

    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "smd.xyz").write_text("")
    (tmp_path / "run" / "smd.json").write_text("{}")
    config = _cfg(
        tmp_path, max_generations=1,
        smd_settings={"inputs": {"smd.xyz": "coord.xyz", "smd.json": "smd.json"}},
    )
    engine = WorkflowEngine(config)
    try:
        engine.run_to_completion()
        assert False, "expected a WaitingForApproval"
    except WaitingForApproval as e:
        assert e.gate_id == "gen0:smd_run"
    smd_dir = tmp_path / "run" / "0" / "tps" / "smd"
    assert (smd_dir / "coord.xyz").exists()

    import shutil

    shutil.copyfile(config.initial_trajectory, smd_dir / "tps.xyz")
    engine.run_to_completion()
    assert engine.is_done()


def test_reftraj_completion_check(tmp_path):
    from dft.reftraj_check import reftraj_missing

    d = tmp_path / "structure_0"
    d.mkdir()
    (d / "coord.xyz").write_text("a\nb\nc\n")
    assert reftraj_missing(d) == "cp2k-pos-1.xyz missing"
    (d / "cp2k-pos-1.xyz").write_text("a\nb\n")  # half-written
    (d / "cp2k-frc-1.xyz").write_text("a\nb\nc\n")
    assert "cp2k-pos-1.xyz has 2 lines" in reftraj_missing(d)
    (d / "cp2k-pos-1.xyz").write_text("a\nb\nc\n")
    assert reftraj_missing(d) is None


def test_config_json_roundtrip_with_new_options(tmp_path):
    a, b = tmp_path / "a.xyz", tmp_path / "b.xyz"
    cfg = _cfg(
        tmp_path, initial_structures=[a, b], n_dft_splits=3, dft_frames_per_split=100,
        cumulative_training=True, selection_mode="sampler", smd_settings={"inputs": {"x": "y"}},
    )
    cfg.to_json(tmp_path / "c.json")
    back = WorkflowConfig.from_json(tmp_path / "c.json")
    assert back.initial_structures == [a, b]
    assert (back.n_dft_splits, back.dft_frames_per_split) == (3, 100)
    assert back.cumulative_training and back.selection_mode == "sampler"
    assert back.smd_settings == {"inputs": {"x": "y"}}


def test_foundation_models_skip_training(tmp_path):
    model = tmp_path / "foundation.model"
    model.write_text('{"seed": 1, "bias": 0.0}')  # mock backends read json models
    config = _cfg(tmp_path, max_generations=1, mlp_settings={"foundation_models": [str(model)]})
    engine = WorkflowEngine(config)
    engine.run_to_completion()
    assert engine.state.generation(0).model_paths == [str(model)]
    assert not (tmp_path / "run" / "0" / "mlp" / "runs").exists()  # nothing trained


def test_smd_auto_run_builds_result_and_fails_loud(tmp_path, monkeypatch=None):
    import subprocess
    import sys
    from core.xyz_io import iter_raw_blocks

    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "smd.json").write_text("{}")
    _make_toy_system(tmp_path / "run" / "smd.xyz", n_frames=1)
    smd = {"inputs": {"smd.xyz": "coord.xyz", "smd.json": "smd.json"}, "auto_run": True,
           "plumed": None, "stride": 2}
    config = _cfg(tmp_path, max_generations=1, smd_settings=smd)
    engine = WorkflowEngine(config)
    real_run = subprocess.run

    def fake_run(cmd, cwd=None, **kw):  # stand-in for `python -m mlp.run_md`
        assert cmd[:3] == [sys.executable, "-m", "mlp.run_md"], cmd
        _make_toy_system(Path(cwd) / "md_out.xyz", n_frames=6)
        return subprocess.CompletedProcess(cmd, 0)

    subprocess.run = fake_run
    try:
        engine.run_to_completion()
    finally:
        subprocess.run = real_run
    assert engine.is_done()
    assert len(iter_raw_blocks(tmp_path / "run" / "0" / "tps" / "smd" / "tps.xyz")) == 3  # 6 frames, stride 2


def test_paper_selection_weights(tmp_path):
    try:
        import numpy as np
    except ImportError:
        return  # numpy-only helper; covered in the real env
    from tps.util.mlp_helper import paper_selection_weights

    m = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [0.0, 0.0, float("nan")]])
    w = paper_selection_weights(m, lam=0.7)
    assert abs(w.sum() - 1) < 1e-12
    # S = [0.25*3, 0.75*3, 0 (+nan->0)] -> w ~ 0.7*S + 0.3, ordering follows uncertainty
    assert w[1] > w[0] > w[2] > 0
    u = paper_selection_weights(m, lam=0.0)
    assert np.allclose(u, 1 / 3)  # lambda=0: uniform sampling


def test_init_md_stage_gates_runs_and_resumes(tmp_path):
    import subprocess
    import sys
    from core.gates import WaitingForApproval
    from core.xyz_io import iter_raw_blocks

    try:
        import numpy  # noqa: F401  (engine imports mlp.init_md defaults, which needs numpy)
    except ImportError:
        return
    ins = []
    for i in range(2):
        f = tmp_path / f"in{i}.xyz"
        _make_toy_system(f, n_frames=1)
        ins.append(str(f))
    im = {"inputs": ins, "model": "mace_mp:medium", "n_frames": 5, "total_steps": 50}
    config = _cfg(tmp_path, max_generations=1, init_md_settings=im)
    config.gates["init_md"] = "manual"
    engine = WorkflowEngine(config)
    try:
        engine.run_to_completion()
        assert False, "expected init_md gate"
    except WaitingForApproval as e:
        assert e.gate_id == "init_md"
    engine.gates.approve("init_md")
    try:  # approved, not auto_run -> external job listing one command per input
        engine.run_to_completion()
        assert False, "expected NeedsExternalJob"
    except WaitingForApproval as e:
        assert e.instructions.count("mlp.init_md") == 2, e.instructions
    # the person runs the MD (stand-in writes 5 frames per input); engine then resumes and labels all of them
    for i in range(2):
        _make_toy_system(tmp_path / "run" / "init_md" / f"init_{i}.xyz", n_frames=5)
    engine.run_to_completion()
    assert engine.is_done()
    labeled = iter_raw_blocks(Path(engine.state.generation(0).labeled_file))
    assert len(labeled) == 10  # 2 inputs x 5 frames, one structure each
    assert (tmp_path / "run" / "0" / "dft").exists()
    # settings file for the runner was written
    import json
    assert json.loads((tmp_path / "run" / "init_md" / "init_md.json").read_text())["n_frames"] == 5
