"""
conftest.py — Stubs all heavy deps + shared fixtures.

All heavy libraries are injected into sys.modules as MagicMocks at module
load time, before any production code is imported.  Tests therefore run with
no GPU, no CP2K, no MACE, no OpenMM installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

# ---------------------------------------------------------------------------
# Stub heavy deps — must happen before any mlpath module is imported
# ---------------------------------------------------------------------------

_HEAVY = [
    "openmm", "openmm.app", "openmm.app.element", "openmm.app.topology",
    "openmmtools", "openmmtools.integrators",
    "openmmml",
    "simtk", "simtk.unit",
    "mace", "mace.calculators", "mace.cli", "mace.cli.run_train",
    "fairchem", "fairchem.core",
    "fairchem.data", "fairchem.data.oc", "fairchem.data.oc.core",
    "openpathsampling",
    "openpathsampling.engines",
    "openpathsampling.engines.openmm",
    "openpathsampling.experimental",
    "openpathsampling.experimental.storage",
    "mdtraj",
    "MDAnalysis",
    "MDAnalysis.lib",
    "MDAnalysis.lib.distances",
    "torch",
    "cpoker", "cpoker.iparser", "cpoker.genfun",
    "ase.calculators.plumed",
    "aimmd",
]
for _dep in _HEAVY:
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

# simtk.unit: neutral numeric multipliers
_u = sys.modules["simtk.unit"]
for _a in ["nanometer", "kelvin", "picosecond", "femtosecond",
           "BOLTZMANN_CONSTANT_kB", "bar", "atmosphere", "mol", "kJ", "fs"]:
    setattr(_u, _a, 1.0)

# MDAnalysis distance functions — real numpy implementations
import MDAnalysis.lib.distances as _dist

def _distance_array(A, B):
    A = np.atleast_2d(A).reshape(-1, 3)
    B = np.atleast_2d(B).reshape(-1, 3)
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

_dist.distance_array  = _distance_array
_dist.capped_distance = lambda *a, **kw: (np.empty((0, 2), int), np.empty(0))

# OPS experimental storage
_exp = sys.modules["openpathsampling.experimental.storage"]
_exp.monkey_patch_all = lambda x: x
_exp.Storage = MagicMock()

# OPS top-level
_ops = sys.modules["openpathsampling"]
for _n in ["PathMover", "PathSampling", "TPSNetwork", "MoveScheme",
           "CVDefinedVolume", "CoordinateFunctionCV", "FunctionCV",
           "shooting", "strategies"]:
    setattr(_ops, _n, MagicMock())

# VVVRIntegrator
_vi = MagicMock()
_vi.return_value.getTemperature.return_value = 300.0
_vi.return_value.getStepSize.return_value    = 0.001
_vi.return_value.setConstraintTolerance      = MagicMock()
sys.modules["openmmtools.integrators"].VVVRIntegrator = _vi

# ---------------------------------------------------------------------------
# Make mlpath/ importable as a flat namespace (bypass __init__ re-exports)
# ---------------------------------------------------------------------------
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


# ---------------------------------------------------------------------------
# Shared helpers (used both in fixtures and in test files)
# ---------------------------------------------------------------------------

def make_atoms(n: int = 3) -> Atoms:
    symbols = (["O", "H", "H"] + ["H"] * max(0, n - 3))[:n]
    return Atoms(
        symbols=symbols,
        positions=np.random.default_rng(0).random((n, 3)) * 3.0,
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def make_extxyz(path: Path, n_frames: int = 5, n_atoms: int = 3) -> Path:
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(n_frames):
        a = make_atoms(n_atoms)
        a.info["energy_cp2k"] = float(rng.standard_normal())
        a.arrays["forces_cp2k"] = rng.standard_normal((n_atoms, 3))
        frames.append(a)
    write(str(path), frames)
    return path


def add_committee_stats(atoms: Atoms, n_models: int = 3) -> Atoms:
    rng = np.random.default_rng(7)
    n   = len(atoms)
    energies = rng.standard_normal(n_models)
    forces   = rng.standard_normal((n_models, n, 3))
    for i, e in enumerate(energies):
        atoms.info[f"energy_m{i + 1}"] = float(e)
    atoms.info["energy_variance"]        = float(np.var(energies))
    atoms.info["force_variance"]         = float(np.var(forces, axis=0).mean())
    atoms.arrays["forces"]               = forces[0]
    atoms.arrays["force_var_per_atom"]   = np.var(forces, axis=0).mean(axis=1)
    return atoms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workdir(tmp_path):
    import os
    orig = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(orig)


@pytest.fixture
def fake_atoms():
    return make_atoms(3)


@pytest.fixture
def fake_committee_atoms():
    return add_committee_stats(make_atoms(3))


@pytest.fixture
def fake_xyz_file(tmp_workdir):
    return make_extxyz(tmp_workdir / "data.xyz", n_frames=10)


@pytest.fixture
def fake_extdata_xyz(tmp_workdir):
    return make_extxyz(tmp_workdir / "extdata.xyz", n_frames=8)


@pytest.fixture
def fake_tps_json(tmp_workdir):
    cfg = {
        "system_name": "test",
        "platform_name": "CPU",
        "top_file": "top.pdb",
        "mace_model_path": ["model.model"],
        "UNITS": {"size_unit": "nanometer", "temp_unit": "kelvin",
                  "time_unit": "picosecond"},
        "Integrator_settings": {
            "temperature": 330, "dt": 0.001,
            "friction": 1.0, "constant_variable": "V",
        },
        "TPS_settings": {"n_frames_max": 1000, "n_steps_per_frame": 10},
        "CV": {
            "my_cv": {
                "type": "bond",
                "params": {"group_A_atoms": [0], "group_B_atoms": [1]},
                "bound":   [0.0, 1.5],
                "unbound": [2.0, 5.0],
            }
        },
        "tot_mc_steps": 10,
        "save_every": 5,
        "num_select": 50,
        "selection_indices": [0, 1, 2],
        "local_range": 5,
        "init_path": "tps.xyz",
        "save_rejected": False,
    }
    p = tmp_workdir / "tps.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def fake_workflow_config(tmp_workdir):
    cfg = {
        "workdir": str(tmp_workdir),
        "max_generations": 2,
        "structures": ["init/coord.xyz"],
        "top_file": "top.pdb",
        "dft": {
            "engine": "cp2k",
            "input_files": ["cp2k.inp"],
            "settings": {"command": "cp2k"},
            "workdir": "single_reftraj",
        },
        "mlp": {
            "config_file": "mlp.yaml",
            "E0s": {"1": -13.5, "8": -428.9},
            "seeds": [123],
            "test_fraction": 0.25,
        },
        "tps": {"config_file": "tps.json"},
        "auto_confirm": {
            "remove_pause": True,
            "continue_generation": True,
            "run_smd": False,
        },
    }
    p = tmp_workdir / "workflow.json"
    p.write_text(json.dumps(cfg))
    (tmp_workdir / "init").mkdir(exist_ok=True)
    return p


@pytest.fixture
def mock_mace_calculator():
    c = MagicMock()
    c.get_potential_energy.return_value = -42.0
    c.get_forces.return_value = np.zeros((3, 3))
    c.calculate.return_value  = {"energy": -42.0, "forces": np.zeros((3, 3))}
    return c


# ---------------------------------------------------------------------------
# After all production modules are imported, patch distance_array directly
# into util.cv's namespace (it was bound at import time before we could fix it)
# ---------------------------------------------------------------------------
def pytest_configure(config):
    """Called after all conftest.py files are loaded — safe to patch modules."""
    try:
        import util.cv as _cv_mod
        _cv_mod.distance_array = _distance_array
    except Exception:
        pass
